import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 2. LOAD FINAL LEAKAGE-SAFE DATA
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

TARGET = "dn_within_365d"

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nPatients:", len(df))
print("DN positive:", int(df[TARGET].sum()))
print("DN negative:", int((df[TARGET] == 0).sum()))
print("Number of predictors:", len(FEATURES))

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 3. MATRICES
# ============================================================

X_train = train_df[FEATURES].astype(float).values
X_val = val_df[FEATURES].astype(float).values
X_test = test_df[FEATURES].astype(float).values

y_train = train_df[TARGET].astype(int).values
y_val = val_df[TARGET].astype(int).values
y_test = test_df[TARGET].astype(int).values

test_ids = test_df["subject_id"].values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

positive = int(y_train.sum())
negative = int((y_train == 0).sum())

class_ratio = negative / positive

print("\nTraining positives:", positive)
print("Training negatives:", negative)
print("Positive class weight:", round(class_ratio, 4))


# ============================================================
# 4. METRICS
# ============================================================

def select_threshold(y_true, probs):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.05,
        0.951,
        0.005
    ):

        preds = (
            probs >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            preds,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def evaluate(
    model_name,
    val_probs,
    test_probs
):

    threshold = select_threshold(
        y_val,
        val_probs
    )

    preds = (
        test_probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        preds,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    result = {
        "Model": model_name,
        "N_test": len(y_test),
        "AUROC": roc_auc_score(
            y_test,
            test_probs
        ),
        "AUPRC": average_precision_score(
            y_test,
            test_probs
        ),
        "Accuracy": accuracy_score(
            y_test,
            preds
        ),
        "Precision": precision_score(
            y_test,
            preds,
            zero_division=0
        ),
        "Recall_Sensitivity": recall_score(
            y_test,
            preds,
            zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(
            y_test,
            preds,
            zero_division=0
        ),
        "Brier": brier_score_loss(
            y_test,
            test_probs
        ),
        "Threshold": threshold,
        "Val_AUROC": roc_auc_score(
            y_val,
            val_probs
        ),
        "Val_AUPRC": average_precision_score(
            y_val,
            val_probs
        )
    }

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return result


results = []
prediction_df = pd.DataFrame({
    "subject_id": test_ids,
    "label": y_test
})


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

logistic = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED
)

logistic.fit(
    X_train_scaled,
    y_train
)

val_probs = logistic.predict_proba(
    X_val_scaled
)[:, 1]

test_probs = logistic.predict_proba(
    X_test_scaled
)[:, 1]

results.append(
    evaluate(
        "Logistic Regression 365d",
        val_probs,
        test_probs
    )
)

prediction_df[
    "logistic_probability"
] = test_probs


# ============================================================
# 6. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

val_probs = rf.predict_proba(
    X_val
)[:, 1]

test_probs = rf.predict_proba(
    X_test
)[:, 1]

results.append(
    evaluate(
        "Random Forest 365d",
        val_probs,
        test_probs
    )
)

prediction_df[
    "random_forest_probability"
] = test_probs


# ============================================================
# 7. XGBOOST
# ============================================================

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=class_ratio,
    random_state=SEED,
    n_jobs=4
)

xgb.fit(
    X_train,
    y_train
)

val_probs = xgb.predict_proba(
    X_val
)[:, 1]

test_probs = xgb.predict_proba(
    X_test
)[:, 1]

results.append(
    evaluate(
        "XGBoost 365d",
        val_probs,
        test_probs
    )
)

prediction_df[
    "xgboost_probability"
] = test_probs


# ============================================================
# 8. PYTORCH MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


Xtr = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

ytr = torch.tensor(
    y_train,
    dtype=torch.float32
)

train_loader = DataLoader(
    TensorDataset(Xtr, ytr),
    batch_size=128,
    shuffle=True
)

mlp = MLP(
    len(FEATURES)
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [class_ratio],
        dtype=torch.float32,
        device=DEVICE
    )
)

optimizer = torch.optim.Adam(
    mlp.parameters(),
    lr=1e-3
)

best_val_auc = -1
best_state = None
patience = 5
bad_epochs = 0

Xv_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32,
    device=DEVICE
)

for epoch in range(1, 51):

    mlp.train()
    running_loss = 0.0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = mlp(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    mlp.eval()

    with torch.no_grad():

        val_probs = torch.sigmoid(
            mlp(Xv_tensor)
        ).cpu().numpy()

    val_auc = roc_auc_score(
        y_val,
        val_probs
    )

    print(
        f"MLP Epoch {epoch:02d} | "
        f"Loss {running_loss / len(train_loader):.4f} | "
        f"Val AUROC {val_auc:.4f}"
    )

    if val_auc > best_val_auc:

        best_val_auc = val_auc

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in mlp.state_dict().items()
        }

        bad_epochs = 0

    else:

        bad_epochs += 1

        if bad_epochs >= patience:
            print("MLP early stopping")
            break


mlp.load_state_dict(best_state)
mlp.to(DEVICE)
mlp.eval()

Xt_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

with torch.no_grad():

    val_probs = torch.sigmoid(
        mlp(Xv_tensor)
    ).cpu().numpy()

    test_probs = torch.sigmoid(
        mlp(Xt_tensor)
    ).cpu().numpy()

results.append(
    evaluate(
        "MLP 365d",
        val_probs,
        test_probs
    )
)

prediction_df[
    "mlp_probability"
] = test_probs

torch.save(
    best_state,
    "mlp_365d_best.pt"
)


# ============================================================
# 9. SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "structured_baselines_365d_results.csv",
    index=False
)

prediction_df.to_csv(
    "structured_baselines_365d_predictions.csv",
    index=False
)

print("\nSaved: structured_baselines_365d_results.csv")
print("Saved: structured_baselines_365d_predictions.csv")
print("Saved: mlp_365d_best.pt")
