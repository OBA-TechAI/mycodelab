import os
import shutil
import random

import numpy as np
import pandas as pd
import torch
import transtab

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

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4
PATIENCE = 5

CHECKPOINT_DIR = "./transtab_365d_control_ckpt"

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)
print("TransTab version:", transtab.version)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. LOAD LEAKAGE-SAFE DATA
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
    int(
        (df["dn_within_365d"] == 0).sum()
    )
)


# ============================================================
# 3. FEATURES
# ============================================================

NUMERICAL_COLUMNS = [
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

CATEGORICAL_COLUMNS = []
BINARY_COLUMNS = []

print(
    "\nNumber of numerical features:",
    len(NUMERICAL_COLUMNS)
)


# ============================================================
# 4. FIXED PATIENT SPLITS
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

print(
    "Training positives:",
    int(train_df["dn_within_365d"].sum())
)

print(
    "Training negatives:",
    int(
        (
            train_df["dn_within_365d"]
            == 0
        ).sum()
    )
)


# ============================================================
# 5. TRAIN-ONLY STANDARDIZATION
# ============================================================

scaler = StandardScaler()

train_df[
    NUMERICAL_COLUMNS
] = scaler.fit_transform(
    train_df[
        NUMERICAL_COLUMNS
    ]
)

val_df[
    NUMERICAL_COLUMNS
] = scaler.transform(
    val_df[
        NUMERICAL_COLUMNS
    ]
)

test_df[
    NUMERICAL_COLUMNS
] = scaler.transform(
    test_df[
        NUMERICAL_COLUMNS
    ]
)


# ============================================================
# 6. CREATE TRANSTAB INPUTS
# ============================================================

X_train = (
    train_df[
        NUMERICAL_COLUMNS
    ]
    .astype(float)
    .reset_index(drop=True)
)

y_train = (
    train_df[
        "dn_within_365d"
    ]
    .astype(int)
    .reset_index(drop=True)
)

X_val = (
    val_df[
        NUMERICAL_COLUMNS
    ]
    .astype(float)
    .reset_index(drop=True)
)

y_val = (
    val_df[
        "dn_within_365d"
    ]
    .astype(int)
    .reset_index(drop=True)
)

X_test = (
    test_df[
        NUMERICAL_COLUMNS
    ]
    .astype(float)
    .reset_index(drop=True)
)

y_test = (
    test_df[
        "dn_within_365d"
    ]
    .astype(int)
    .reset_index(drop=True)
)

trainset = (
    X_train,
    y_train
)

valset = (
    X_val,
    y_val
)


# ============================================================
# 7. BUILD OFFICIAL TRANSTAB CLASSIFIER
# ============================================================

model = transtab.build_classifier(
    categorical_columns=CATEGORICAL_COLUMNS,
    numerical_columns=NUMERICAL_COLUMNS,
    binary_columns=BINARY_COLUMNS,

    num_class=2,

    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0.1,
    ffn_dim=256,
    activation="relu",

    device=DEVICE
)

print(
    "\nModel:",
    type(model).__name__
)


# ============================================================
# 8. CLEAN OUTPUT DIRECTORY
# ============================================================

if os.path.exists(
    CHECKPOINT_DIR
):
    shutil.rmtree(
        CHECKPOINT_DIR
    )


# ============================================================
# 9. TRAIN TRANSTAB
# ============================================================

print(
    "\nStarting TransTab training..."
)

transtab.train(
    model,
    trainset,
    valset,

    num_epoch=EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=256,

    lr=LR,
    weight_decay=1e-4,
    patience=PATIENCE,

    eval_metric="auc",

    output_dir=CHECKPOINT_DIR,

    num_workers=0,

    # Handle strong class imbalance
    balance_sample=True,

    # Restore best validation model
    load_best_at_last=True
)

print(
    "\nTraining completed"
)


# ============================================================
# 10. VALIDATION PREDICTIONS
# ============================================================

val_prob = transtab.predict(
    model,
    X_val,
    eval_batch_size=256
)

val_prob = np.asarray(
    val_prob
).reshape(-1)

val_y = y_val.values


print(
    "\nValidation AUROC:",
    round(
        roc_auc_score(
            val_y,
            val_prob
        ),
        6
    )
)

print(
    "Validation AUPRC:",
    round(
        average_precision_score(
            val_y,
            val_prob
        ),
        6
    )
)


# ============================================================
# 11. SELECT THRESHOLD ON VALIDATION ONLY
# ============================================================

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
# 12. TEST PREDICTIONS
# ============================================================

test_prob = transtab.predict(
    model,
    X_test,
    eval_batch_size=256
)

test_prob = np.asarray(
    test_prob
).reshape(-1)

test_y = y_test.values

test_pred = (
    test_prob
    >= best_threshold
).astype(int)


# ============================================================
# 13. TEST METRICS
# ============================================================

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
        "TransTab 365d Control",

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
        best_threshold
}


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)

print(
    "LEAKAGE-SAFE TRANSTAB CONTROL RESULTS"
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
# 15. SAVE RESULTS
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    "transtab_365d_control_results.csv",
    index=False
)

pd.DataFrame({

    "subject_id":
        test_df[
            "subject_id"
        ].astype(int).values,

    "y_true":
        test_y,

    "y_prob":
        test_prob,

    "y_pred":
        test_pred

}).to_csv(
    "transtab_365d_control_predictions.csv",
    index=False
)


print(
    "\nSaved: transtab_365d_control_results.csv"
)

print(
    "Saved: transtab_365d_control_predictions.csv"
)

print(
    "Checkpoint directory:",
    CHECKPOINT_DIR
)
