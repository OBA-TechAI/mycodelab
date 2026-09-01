import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
MAX_LEN = 256
MAX_VOCAB = 14000

HIDDEN = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
PATIENCE = 4

MAX_AGE = 120
MAX_VISITS = 64

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
# 2. LOAD LEAKAGE-SAFE DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_365d_age.parquet"
)

cohort = pd.read_parquet(
    "cohort_labels_timeaware_365d.parquet"
)[[
    "subject_id",
    "dn_within_365d"
]]

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

cohort = cohort.merge(
    splits,
    on="subject_id",
    how="inner"
)

seq["subject_id"] = seq["subject_id"].astype(int)
cohort["subject_id"] = cohort["subject_id"].astype(int)

print("\nPatients:", len(cohort))
print(
    "DN positive:",
    int(cohort["dn_within_365d"].sum())
)
print(
    "DN negative:",
    int((cohort["dn_within_365d"] == 0).sum())
)


# ============================================================
# 3. FIXED SPLITS
# ============================================================

train_ids = cohort.loc[
    cohort["split"] == "train",
    "subject_id"
].tolist()

val_ids = cohort.loc[
    cohort["split"] == "validation",
    "subject_id"
].tolist()

test_ids = cohort.loc[
    cohort["split"] == "test",
    "subject_id"
].tolist()

print("\nTrain:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# 4. TRAIN-ONLY VOCABULARY
# ============================================================

train_codes = seq[
    seq["subject_id"].isin(train_ids)
]["icd_token"].value_counts()

top_codes = train_codes.index[
    : MAX_VOCAB - 4
].tolist()

code_to_id = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<VS>": 2,
    "<VE>": 3
}

for i, code in enumerate(
    top_codes,
    start=4
):
    code_to_id[code] = i

VOCAB_SIZE = len(code_to_id)

print(
    "\nVocabulary size:",
    VOCAB_SIZE
)


# ============================================================
# 5. LABEL LOOKUP
# ============================================================

label_map = dict(
    zip(
        cohort["subject_id"],
        cohort["dn_within_365d"]
    )
)


# ============================================================
# 6. BUILD CEHR-BERT-STYLE SEQUENCES
# ============================================================

patient_data = {}

truncated = 0

seq = seq.sort_values(
    [
        "subject_id",
        "visit_number",
        "code_rank"
    ]
)

for sid, patient_df in seq.groupby(
    "subject_id"
):

    tokens = []
    ages = []
    visit_types = []

    for visit_number, visit_df in patient_df.groupby(
        "visit_number",
        sort=True
    ):

        visit_df = visit_df.sort_values(
            "code_rank"
        )

        age = int(
            visit_df[
                "visit_age"
            ].iloc[0]
        )

        age = max(
            0,
            min(MAX_AGE, age)
        )

        visit_type = int(
            (int(visit_number) - 1) % 2
        )

        # Visit-start token
        tokens.append(
            code_to_id["<VS>"]
        )
        ages.append(age)
        visit_types.append(
            visit_type
        )

        # Diagnosis concepts
        for code in visit_df[
            "icd_token"
        ]:

            tokens.append(
                code_to_id.get(
                    code,
                    code_to_id["<UNK>"]
                )
            )

            ages.append(age)
            visit_types.append(
                visit_type
            )

        # Visit-end token
        tokens.append(
            code_to_id["<VE>"]
        )

        ages.append(age)
        visit_types.append(
            visit_type
        )

    # Keep most recent history
    if len(tokens) > MAX_LEN:

        truncated += 1

        tokens = tokens[-MAX_LEN:]
        ages = ages[-MAX_LEN:]
        visit_types = (
            visit_types[-MAX_LEN:]
        )

    patient_data[
        int(sid)
    ] = {
        "codes": tokens,
        "ages": ages,
        "visit_types": visit_types
    }


print(
    "Patients truncated to",
    MAX_LEN,
    "tokens:",
    truncated
)


# ============================================================
# 7. DATASET
# ============================================================

class CEHRBERTDataset(Dataset):

    def __init__(
        self,
        patient_ids
    ):
        self.patient_ids = (
            patient_ids
        )

    def __len__(self):
        return len(
            self.patient_ids
        )

    def __getitem__(
        self,
        idx
    ):

        sid = int(
            self.patient_ids[idx]
        )

        p = patient_data[sid]

        codes = list(
            p["codes"]
        )

        ages = list(
            p["ages"]
        )

        visit_types = list(
            p["visit_types"]
        )

        length = len(codes)

        positions = list(
            range(length)
        )

        pad = (
            MAX_LEN - length
        )

        codes += [0] * pad
        ages += [0] * pad
        visit_types += [0] * pad
        positions += [0] * pad

        return {

            "subject_id":
                sid,

            "codes":
                torch.tensor(
                    codes,
                    dtype=torch.long
                ),

            "ages":
                torch.tensor(
                    ages,
                    dtype=torch.long
                ),

            "visit_types":
                torch.tensor(
                    visit_types,
                    dtype=torch.long
                ),

            "positions":
                torch.tensor(
                    positions,
                    dtype=torch.long
                ),

            "label":
                torch.tensor(
                    float(
                        label_map[sid]
                    ),
                    dtype=torch.float32
                )
        }


train_loader = DataLoader(
    CEHRBERTDataset(
        train_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    CEHRBERTDataset(
        val_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    CEHRBERTDataset(
        test_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. CEHR-BERT-STYLE TEMPORAL ABLATION CONTROL
# ============================================================

class CEHRBERTControl(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.code_embedding = (
            nn.Embedding(
                VOCAB_SIZE,
                HIDDEN,
                padding_idx=0
            )
        )

        # Age is retained in both
        # control and future time-aware model
        self.age_embedding = (
            nn.Embedding(
                MAX_AGE + 1,
                HIDDEN
            )
        )

        self.visit_type_embedding = (
            nn.Embedding(
                2,
                HIDDEN
            )
        )

        self.position_embedding = (
            nn.Embedding(
                MAX_LEN,
                HIDDEN
            )
        )

        self.layer_norm = (
            nn.LayerNorm(
                HIDDEN
            )
        )

        self.dropout = (
            nn.Dropout(
                DROPOUT
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=HIDDEN,
                nhead=N_HEADS,
                dim_feedforward=(
                    HIDDEN * 4
                ),
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=N_LAYERS
            )
        )

        self.classifier = (
            nn.Sequential(
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
        )

    def forward(
        self,
        codes,
        ages,
        visit_types,
        positions
    ):

        padding_mask = (
            codes.eq(0)
        )

        x = (
            self.code_embedding(
                codes
            )
            +
            self.age_embedding(
                ages
            )
            +
            self.visit_type_embedding(
                visit_types
            )
            +
            self.position_embedding(
                positions
            )
        )

        x = self.layer_norm(x)

        x = self.dropout(x)

        x = self.transformer(
            x,
            src_key_padding_mask=(
                padding_mask
            )
        )

        valid = (
            (~padding_mask)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            (x * valid).sum(
                dim=1
            )
            /
            valid.sum(
                dim=1
            ).clamp(
                min=1.0
            )
        )

        logits = (
            self.classifier(
                pooled
            )
            .squeeze(-1)
        )

        return logits


model = (
    CEHRBERTControl()
    .to(DEVICE)
)

print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 9. CLASS WEIGHT
# ============================================================

train_labels = cohort[
    cohort["split"]
    == "train"
]["dn_within_365d"]

n_pos = int(
    train_labels.sum()
)

n_neg = int(
    (
        train_labels == 0
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

criterion = (
    nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )
)

optimizer = (
    torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
)


# ============================================================
# 10. PREDICTION
# ============================================================

def predict(loader):

    model.eval()

    subject_ids = []
    labels = []
    probabilities = []

    with torch.no_grad():

        for batch in loader:

            logits = model(

                batch[
                    "codes"
                ].to(DEVICE),

                batch[
                    "ages"
                ].to(DEVICE),

                batch[
                    "visit_types"
                ].to(DEVICE),

                batch[
                    "positions"
                ].to(DEVICE)
            )

            probs = (
                torch.sigmoid(
                    logits
                )
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
        np.asarray(
            subject_ids
        ),
        np.asarray(
            labels,
            dtype=int
        ),
        np.asarray(
            probabilities
        )
    )


# ============================================================
# 11. TRAINING
# ============================================================

BEST_MODEL = (
    "cehrbert_365d_control_best.pt"
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

        logits = model(

            batch[
                "codes"
            ].to(DEVICE),

            batch[
                "ages"
            ].to(DEVICE),

            batch[
                "visit_types"
            ].to(DEVICE),

            batch[
                "positions"
            ].to(DEVICE)
        )

        labels = batch[
            "label"
        ].to(DEVICE)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

    _, val_y, val_prob = (
        predict(
            val_loader
        )
    )

    val_auc = (
        roc_auc_score(
            val_y,
            val_prob
        )
    )

    val_auprc = (
        average_precision_score(
            val_y,
            val_prob
        )
    )

    mean_loss = (
        running_loss
        /
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

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping at epoch",
                epoch
            )

            break


print(
    "\nBest validation AUROC:",
    round(
        best_auc,
        4
    ),
    "at epoch",
    best_epoch
)


# ============================================================
# 12. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)


# ============================================================
# 13. VALIDATION THRESHOLD
# ============================================================

_, val_y, val_prob = (
    predict(
        val_loader
    )
)

best_threshold = 0.50
best_f1 = -1

for threshold in np.arange(
    0.05,
    0.951,
    0.005
):

    pred = (
        val_prob
        >= threshold
    ).astype(int)

    score = f1_score(
        val_y,
        pred,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score

        best_threshold = (
            float(
                threshold
            )
        )


print(
    "Selected validation threshold:",
    round(
        best_threshold,
        3
    )
)


# ============================================================
# 14. TEST EVALUATION
# ============================================================

test_subjects, test_y, test_prob = (
    predict(
        test_loader
    )
)

test_pred = (
    test_prob
    >= best_threshold
).astype(int)

tn, fp, fn, tp = (
    confusion_matrix(
        test_y,
        test_pred,
        labels=[0, 1]
    ).ravel()
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

results = {

    "Model":
        "CEHR-BERT-style 365d Temporal-Ablation Control",

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
# 15. PRINT
# ============================================================

print("\n")
print("=" * 65)
print(
    "LEAKAGE-SAFE CEHR-BERT-STYLE CONTROL RESULTS"
)
print("=" * 65)

for key, value in (
    results.items()
):

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
# 16. SAVE
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    "cehrbert_365d_control_results.csv",
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
    "cehrbert_365d_control_predictions.csv",
    index=False
)

print(
    "\nSaved: cehrbert_365d_control_results.csv"
)

print(
    "Saved: cehrbert_365d_control_predictions.csv"
)

print(
    "Saved:",
    BEST_MODEL
)
