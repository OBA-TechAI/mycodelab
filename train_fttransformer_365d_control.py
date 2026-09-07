import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

HIDDEN = 128
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_parquet(
    "structured_features_365d_safe.parquet"
)

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

df = df.merge(
    splits,
    on="subject_id",
    how="inner"
)

print("\nPatients:", len(df))
print(
    "DN positive:",
    int(df["dn_within_365d"].sum())
)

print(
    "DN negative:",
    int((df["dn_within_365d"] == 0).sum())
)


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "prior_dx_count",
    "prior_unique_dx_count",
    "prior_visit_count",
    "age_at_last_prior_visit",
    "mean_codes_per_visit",
    "max_codes_per_visit",
    "mean_unique_codes_per_visit",
    "prior_icd9_count",
    "prior_icd10_count"
]

print(
    "\nNumber of features:",
    len(FEATURES)
)


# ============================================================
# 4. FIXED SPLITS
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 5. SCALE USING TRAINING DATA ONLY
# ============================================================

scaler = StandardScaler()

train_df[FEATURES] = scaler.fit_transform(
    train_df[FEATURES]
)

val_df[FEATURES] = scaler.transform(
    val_df[FEATURES]
)

test_df[FEATURES] = scaler.transform(
    test_df[FEATURES]
)


# ============================================================
# 6. DATASET
# ============================================================

class TabularDataset(Dataset):

    def __init__(self, dataframe):

        self.subject_ids = (
            dataframe["subject_id"]
            .astype(int)
            .values
        )

        self.x = (
            dataframe[FEATURES]
            .astype(np.float32)
            .values
        )

        self.y = (
            dataframe["dn_within_365d"]
            .astype(np.float32)
            .values
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return {
            "subject_id":
                int(self.subject_ids[idx]),

            "features":
                torch.tensor(
                    self.x[idx],
                    dtype=torch.float32
                ),

            "label":
                torch.tensor(
                    self.y[idx],
                    dtype=torch.float32
                )
        }


train_loader = DataLoader(
    TabularDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TabularDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TabularDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. FT-TRANSFORMER
# ============================================================

class FTTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.n_features = len(FEATURES)

        # Each numerical variable becomes a token
        self.feature_weight = nn.Parameter(
            torch.randn(
                self.n_features,
                HIDDEN
            ) * 0.02
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(
                self.n_features,
                HIDDEN
            )
        )

        self.feature_embedding = nn.Embedding(
            self.n_features,
            HIDDEN
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                HIDDEN
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS
        )

        self.norm = nn.LayerNorm(
            HIDDEN
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                HIDDEN,
                1
            )
        )


    def forward(self, x):

        # x:
        # batch x number_of_features

        tokens = (
            x.unsqueeze(-1)
            * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )

        feature_ids = torch.arange(
            self.n_features,
            device=x.device
        )

        tokens = (
            tokens
            + self.feature_embedding(
                feature_ids
            ).unsqueeze(0)
        )

        cls = self.cls_token.expand(
            x.size(0),
            -1,
            -1
        )

        tokens = torch.cat(
            [cls, tokens],
            dim=1
        )

        output = self.transformer(
            tokens
        )

        cls_output = output[:, 0]

        cls_output = self.norm(
            cls_output
        )

        logits = self.classifier(
            cls_output
        ).squeeze(-1)

        return logits


model = FTTransformer().to(
    DEVICE
)

print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 8. CLASS WEIGHT
# ============================================================

n_pos = int(
    train_df[
        "dn_within_365d"
    ].sum()
)

n_neg = int(
    (
        train_df[
            "dn_within_365d"
        ] == 0
    ).sum()
)

pos_weight_value = (
    n_neg / n_pos
)

print(
    "Training positives:",
    n_pos
)

print(
    "Training negatives:",
    n_neg
)

print(
    "Positive class weight:",
    round(
        pos_weight_value,
        4
    )
)

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# 9. PREDICTION
# ============================================================

def predict(loader):

    model.eval()

    subject_ids = []
    labels = []
    probabilities = []

    with torch.no_grad():

        for batch in loader:

            x = batch[
                "features"
            ].to(DEVICE)

            logits = model(x)

            probs = torch.sigmoid(
                logits
            )

            subject_ids.extend(
                batch[
                    "subject_id"
                ].cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch[
                    "label"
                ].cpu()
                .numpy()
                .tolist()
            )

            probabilities.extend(
                probs.cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(subject_ids),
        np.asarray(
            labels,
            dtype=int
        ),
        np.asarray(
            probabilities
        )
    )


# ============================================================
# 10. TRAIN
# ============================================================

BEST_MODEL = (
    "fttransformer_365d_control_best.pt"
)

best_auc = -np.inf
best_epoch = 0
patience_counter = 0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for batch in train_loader:

        optimizer.zero_grad()

        x = batch[
            "features"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

    _, val_y, val_prob = predict(
        val_loader
    )

    val_auc = roc_auc_score(
        val_y,
        val_prob
    )

    val_auprc = average_precision_score(
        val_y,
        val_prob
    )

    mean_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {mean_loss:.4f} | "
        f"Val AUROC {val_auc:.4f} | "
        f"Val AUPRC {val_auprc:.4f}"
    )

    if val_auc > best_auc:

        best_auc = val_auc
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping at epoch",
                epoch
            )

            break


print(
    "\nBest validation AUROC:",
    round(best_auc, 4),
    "at epoch",
    best_epoch
)


# ============================================================
# 11. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)


# ============================================================
# 12. VALIDATION THRESHOLD
# ============================================================

_, val_y, val_prob = predict(
    val_loader
)

best_threshold = 0.50
best_f1 = -1

for threshold in np.arange(
    0.05,
    0.951,
    0.005
):

    pred = (
        val_prob >= threshold
    ).astype(int)

    score = f1_score(
        val_y,
        pred,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score

        best_threshold = float(
            threshold
        )


print(
    "Selected validation threshold:",
    round(
        best_threshold,
        3
    )
)


# ============================================================
# 13. TEST EVALUATION
# ============================================================

test_subjects, test_y, test_prob = predict(
    test_loader
)

test_pred = (
    test_prob >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_y,
    test_pred,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

results = {

    "Model":
        "FT-Transformer 365d Control",

    "N_test":
        len(test_y),

    "AUROC":
        roc_auc_score(
            test_y,
            test_prob
        ),

    "AUPRC":
        average_precision_score(
            test_y,
            test_prob
        ),

    "Accuracy":
        accuracy_score(
            test_y,
            test_pred
        ),

    "Precision":
        precision_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Recall_Sensitivity":
        recall_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Specificity":
        specificity,

    "F1":
        f1_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Brier":
        brier_score_loss(
            test_y,
            test_prob
        ),

    "Threshold":
        best_threshold,

    "Best_Val_AUROC":
        best_auc,

    "Best_Epoch":
        best_epoch
}


# ============================================================
# 14. PRINT
# ============================================================

print("\n")
print("=" * 60)

print(
    "LEAKAGE-SAFE FT-TRANSFORMER CONTROL RESULTS"
)

print("=" * 60)

for key, value in results.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# 15. SAVE
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    "fttransformer_365d_control_results.csv",
    index=False
)

pd.DataFrame({
    "subject_id":
        test_subjects,

    "y_true":
        test_y,

    "y_prob":
        test_prob,

    "y_pred":
        test_pred

}).to_csv(
    "fttransformer_365d_control_predictions.csv",
    index=False
)

print(
    "\nSaved: fttransformer_365d_control_results.csv"
)

print(
    "Saved: fttransformer_365d_control_predictions.csv"
)

print(
    "Saved:",
    BEST_MODEL
)
