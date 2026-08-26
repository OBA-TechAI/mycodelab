# %% [markdown]
# # Longitudinal MIMIC-IV DN Prediction: Four-Model Comparison
# 
# This revised notebook uses the corrected longitudinal cohort:
# 
# - `cohort_labels_longitudinal.parquet`
# - `dx_sequences_longitudinal.parquet`
# - `patient_splits.parquet`
# 
# It trains **ClinicalBERT EHR-to-text, Med-BERT-style, TransTab, and FT-Transformer** on the same patients with usable pre-index history.
# 
# **Important:** the earlier `structured_features.parquet` and `early_features_24h/48h.parquet` are not used here because they were built against the old index admission. The tabular features in this notebook are derived only from pre-index longitudinal diagnosis history, avoiding temporal leakage.
# 

# %%
"""
LONGITUDINAL MIMIC-IV DN RISK PREDICTION
========================================

Models
------
1. ClinicalBERT EHR-to-text
2. Med-BERT-style longitudinal EHR Transformer
3. TransTab
4. FT-Transformer

Required files in DATA_DIR
--------------------------
- cohort_labels_longitudinal.parquet
- dx_sequences_longitudinal.parquet
- patient_splits.parquet

Why this version is different
-----------------------------
The original cohort used the first admission as the index admission, so there
was no pre-index history. This version uses the corrected longitudinal cohort:
    prior admission(s) -> index admission -> DN outcome

For a fair four-model comparison, all four models use the SAME patients with
at least one usable pre-index diagnosis record.

IMPORTANT:
- Do NOT use the old early_features_24h/48h or structured_features files with
  this longitudinal cohort unless you regenerate them using the NEW index_hadm_id.
- This notebook derives leakage-safe features directly from PRE-INDEX ICD history.
"""

# %%
# ============================================================
# 0. INSTALLATION
# Run once in the Python (MIMIC-DN) Jupyter kernel.
# ============================================================

#%pip install pyarrow pandas numpy scikit-learn
#%pip install "transformers<=4.30.0" accelerate
#%pip install pytorch-tabular
#%pip install git+https://github.com/RyanWangZf/transtab.git

# %%
# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

from pathlib import Path
from collections import Counter, defaultdict
import copy
import json
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch

print("=" * 50)
print("HPC GPU CHECK")
print("=" * 50)

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

print("=" * 50)

#raise SystemExit("GPU check completed successfully.")

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42

# If this notebook is in the same folder as the parquet files:
DATA_DIR = Path.cwd()

# Otherwise replace the line above, for example:
# DATA_DIR = Path(
#     r"C:\Users\oluwa\OneDrive\Desktop\CoC Files\Current Literature\DNTransformerLLMsMIMIC\outputs-20260805T024105Z-1-001\outputs"
# )

COHORT_FILE = DATA_DIR / "cohort_labels_longitudinal.parquet"
DX_FILE = DATA_DIR / "dx_sequences_longitudinal.parquet"
SPLIT_FILE = DATA_DIR / "patient_splits.parquet"

ID_COL = "subject_id"
TARGET_COL = "primary_icd_dn_label"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Conservative defaults for CPU.
CLINICALBERT_MAX_LEN = 256
MEDBERT_MAX_LEN = 256

RUN_CLINICALBERT = True
RUN_MEDBERT = False
RUN_TRANSTAB = False
RUN_FTTRANSFORMER = False


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()

print("Device:", DEVICE)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %%
# ============================================================
# 2. LOAD CORRECTED LONGITUDINAL DATA
# ============================================================

for f in [COHORT_FILE, DX_FILE, SPLIT_FILE]:
    if not f.exists():
        raise FileNotFoundError(
            f"Missing file: {f}\n"
            "Either put the notebook in the same folder as the three parquet "
            "files or change DATA_DIR."
        )

cohort = pd.read_parquet(COHORT_FILE)
dx = pd.read_parquet(DX_FILE)
splits = pd.read_parquet(SPLIT_FILE)

print("Longitudinal cohort:", cohort.shape)
print("Diagnosis sequence:", dx.shape)
print("Patient splits:", splits.shape)

print("\nCohort columns:")
print(cohort.columns.tolist())

print("\nDX columns:")
print(dx.columns.tolist())

# %%
# ============================================================
# 3. CREATE ONE COMMON COHORT FOR ALL FOUR MODELS
# ============================================================

required_cohort = {
    ID_COL,
    "index_hadm_id",
    "index_admittime",
    TARGET_COL,
    "n_prior_visits",
    "index_visit_number",
}

missing = required_cohort - set(cohort.columns)

if missing:
    raise ValueError(
        f"cohort_labels_longitudinal.parquet is missing: {missing}"
    )

required_dx = {
    ID_COL,
    "hadm_id",
    "visit_number",
    "icd_code",
    "icd_version",
    "code_rank_within_visit",
}

missing = required_dx - set(dx.columns)

if missing:
    raise ValueError(
        f"dx_sequences_longitudinal.parquet is missing: {missing}"
    )

if "split" not in splits.columns:
    raise ValueError(
        "patient_splits.parquet must contain a 'split' column."
    )

# Patients with at least one usable diagnosis row.
dx_patient_ids = pd.DataFrame(
    {ID_COL: dx[ID_COL].drop_duplicates()}
)

# Keep only patients usable by every model.
model_cohort = (
    cohort
    .merge(dx_patient_ids, on=ID_COL, how="inner")
    .merge(
        splits[[ID_COL, "split"]].drop_duplicates(ID_COL),
        on=ID_COL,
        how="inner",
    )
)

model_cohort["split"] = (
    model_cohort["split"]
    .astype(str)
    .str.lower()
)

model_cohort[TARGET_COL] = (
    model_cohort[TARGET_COL].astype(int)
)

valid_splits = {"train", "validation", "test"}

if not set(model_cohort["split"].unique()).issubset(valid_splits):
    raise ValueError(
        "Unexpected split names found in patient_splits.parquet."
    )

print("\nCOMMON FOUR-MODEL COHORT")
print("Patients:", model_cohort[ID_COL].nunique())

print("\nSplit sizes:")
print(model_cohort["split"].value_counts())

print("\nDN distribution by split:")
print(
    model_cohort
    .groupby("split")[TARGET_COL]
    .agg(["count", "sum", "mean"])
    .rename(
        columns={
            "sum": "DN_positive",
            "mean": "DN_rate",
        }
    )
)

# %%
# ============================================================
# 4. PREPARE LONGITUDINAL ICD TOKENS
# ============================================================

dx_work = dx[
    dx[ID_COL].isin(model_cohort[ID_COL])
].copy()

dx_work["icd_code"] = (
    dx_work["icd_code"]
    .astype(str)
    .str.upper()
    .str.replace(".", "", regex=False)
)

dx_work["icd_version"] = (
    pd.to_numeric(
        dx_work["icd_version"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

dx_work["icd_token"] = (
    "ICD"
    + dx_work["icd_version"].astype(str)
    + "_"
    + dx_work["icd_code"]
)

dx_work["visit_number"] = (
    pd.to_numeric(
        dx_work["visit_number"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

dx_work["code_rank_within_visit"] = (
    pd.to_numeric(
        dx_work["code_rank_within_visit"],
        errors="coerce",
    )
    .fillna(1)
    .astype(int)
)

if "admittime" in dx_work.columns:
    dx_work["admittime"] = pd.to_datetime(
        dx_work["admittime"],
        errors="coerce",
    )

dx_work = dx_work.sort_values(
    [
        ID_COL,
        "visit_number",
        "code_rank_within_visit",
    ]
).reset_index(drop=True)

print("\nDiagnosis rows used:", len(dx_work))
print(
    "Patients with diagnosis history:",
    dx_work[ID_COL].nunique(),
)

# %%
# ============================================================
# 5. BUILD LEAKAGE-SAFE TABULAR FEATURES FROM PRE-INDEX HISTORY
# ============================================================
"""
These variables are derived ONLY from dx_sequences_longitudinal.parquet, which
contains admissions before the index admission.

This means TransTab and FT-Transformer do not rely on the old structured/24h/48h
files tied to the previous index admission.
"""


def starts_with_any(code, prefixes):
    return any(
        str(code).startswith(prefix)
        for prefix in prefixes
    )


def diagnosis_groups(row):
    """
    Convert an ICD code into clinically interpretable prior-history indicators.
    These are predictors, not labels, because the source sequence is pre-index.
    """

    code = str(row["icd_code"])
    version = int(row["icd_version"])

    groups = {
        "prior_diabetes": 0,
        "prior_ckd": 0,
        "prior_hypertension": 0,
        "prior_heart_failure": 0,
        "prior_ischemic_heart_disease": 0,
        "prior_cerebrovascular": 0,
        "prior_obesity": 0,
        "prior_dyslipidemia": 0,
        "prior_diabetic_neuropathy": 0,
    }

    if version == 9:
        groups["prior_diabetes"] = int(
            code.startswith("250")
        )
        groups["prior_ckd"] = int(
            code.startswith("585")
        )
        groups["prior_hypertension"] = int(
            starts_with_any(
                code,
                ("401", "402", "403", "404", "405"),
            )
        )
        groups["prior_heart_failure"] = int(
            code.startswith("428")
        )
        groups["prior_ischemic_heart_disease"] = int(
            starts_with_any(
                code,
                (
                    "410",
                    "411",
                    "412",
                    "413",
                    "414",
                ),
            )
        )
        groups["prior_cerebrovascular"] = int(
            starts_with_any(
                code,
                (
                    "430",
                    "431",
                    "432",
                    "433",
                    "434",
                    "435",
                    "436",
                    "437",
                    "438",
                ),
            )
        )
        groups["prior_obesity"] = int(
            code.startswith("2780")
        )
        groups["prior_dyslipidemia"] = int(
            code.startswith("272")
        )
        groups["prior_diabetic_neuropathy"] = int(
            code.startswith("2506")
        )

    elif version == 10:
        diabetes_prefixes = (
            "E08",
            "E09",
            "E10",
            "E11",
            "E13",
        )

        groups["prior_diabetes"] = int(
            starts_with_any(
                code,
                diabetes_prefixes,
            )
        )

        groups["prior_ckd"] = int(
            code.startswith("N18")
        )

        groups["prior_hypertension"] = int(
            starts_with_any(
                code,
                (
                    "I10",
                    "I11",
                    "I12",
                    "I13",
                    "I15",
                    "I16",
                ),
            )
        )

        groups["prior_heart_failure"] = int(
            code.startswith("I50")
        )

        groups["prior_ischemic_heart_disease"] = int(
            starts_with_any(
                code,
                (
                    "I20",
                    "I21",
                    "I22",
                    "I23",
                    "I24",
                    "I25",
                ),
            )
        )

        groups["prior_cerebrovascular"] = int(
            starts_with_any(
                code,
                (
                    "I60",
                    "I61",
                    "I62",
                    "I63",
                    "I64",
                    "I65",
                    "I66",
                    "I67",
                    "I68",
                    "I69",
                ),
            )
        )

        groups["prior_obesity"] = int(
            code.startswith("E66")
        )

        groups["prior_dyslipidemia"] = int(
            code.startswith("E78")
        )

        # Diabetes with neurologic complications: .4x families.
        groups["prior_diabetic_neuropathy"] = int(
            any(
                code.startswith(prefix + "4")
                for prefix in diabetes_prefixes
            )
        )

    return pd.Series(groups)


clinical_groups = dx_work.apply(
    diagnosis_groups,
    axis=1,
)

dx_enriched = pd.concat(
    [
        dx_work.reset_index(drop=True),
        clinical_groups.reset_index(drop=True),
    ],
    axis=1,
)

# Visit-level code counts.
visit_counts = (
    dx_enriched
    .groupby(
        [ID_COL, "visit_number"],
        as_index=False,
    )
    .agg(
        codes_in_visit=(
            "icd_token",
            "size",
        )
    )
)

visit_summary = (
    visit_counts
    .groupby(ID_COL)
    .agg(
        prior_visit_count=(
            "visit_number",
            "nunique",
        ),
        mean_codes_per_visit=(
            "codes_in_visit",
            "mean",
        ),
        max_codes_per_visit=(
            "codes_in_visit",
            "max",
        ),
    )
    .reset_index()
)

dx_summary = (
    dx_enriched
    .groupby(ID_COL)
    .agg(
        prior_dx_count=(
            "icd_token",
            "size",
        ),
        prior_unique_dx_count=(
            "icd_token",
            "nunique",
        ),
        prior_icd9_count=(
            "icd_version",
            lambda x: int((x == 9).sum()),
        ),
        prior_icd10_count=(
            "icd_version",
            lambda x: int((x == 10).sum()),
        ),
        prior_diabetes=(
            "prior_diabetes",
            "max",
        ),
        prior_ckd=(
            "prior_ckd",
            "max",
        ),
        prior_hypertension=(
            "prior_hypertension",
            "max",
        ),
        prior_heart_failure=(
            "prior_heart_failure",
            "max",
        ),
        prior_ischemic_heart_disease=(
            "prior_ischemic_heart_disease",
            "max",
        ),
        prior_cerebrovascular=(
            "prior_cerebrovascular",
            "max",
        ),
        prior_obesity=(
            "prior_obesity",
            "max",
        ),
        prior_dyslipidemia=(
            "prior_dyslipidemia",
            "max",
        ),
        prior_diabetic_neuropathy=(
            "prior_diabetic_neuropathy",
            "max",
        ),
    )
    .reset_index()
)

# Temporal-history features if admittime exists.
if (
    "admittime" in dx_enriched.columns
    and dx_enriched["admittime"].notna().any()
):
    admission_dates = (
        dx_enriched[
            [
                ID_COL,
                "visit_number",
                "admittime",
            ]
        ]
        .drop_duplicates(
            [ID_COL, "visit_number"]
        )
        .sort_values(
            [ID_COL, "visit_number"]
        )
    )

    temporal_rows = []

    for sid, group in admission_dates.groupby(ID_COL):
        dates = (
            group["admittime"]
            .dropna()
            .sort_values()
            .tolist()
        )

        if not dates:
            history_span = 0.0
            last_gap = 0.0

        else:
            history_span = (
                dates[-1] - dates[0]
            ).days

            if len(dates) >= 2:
                last_gap = (
                    dates[-1] - dates[-2]
                ).days
            else:
                last_gap = 0.0

        temporal_rows.append(
            {
                ID_COL: sid,
                "history_span_days": max(
                    float(history_span),
                    0.0,
                ),
                "last_prior_visit_gap_days": max(
                    float(last_gap),
                    0.0,
                ),
            }
        )

    temporal_summary = pd.DataFrame(
        temporal_rows
    )

else:
    temporal_summary = (
        model_cohort[[ID_COL]]
        .copy()
    )

    temporal_summary[
        "history_span_days"
    ] = 0.0

    temporal_summary[
        "last_prior_visit_gap_days"
    ] = 0.0


tabular = (
    model_cohort[
        [
            ID_COL,
            TARGET_COL,
            "split",
        ]
    ]
    .merge(
        visit_summary,
        on=ID_COL,
        how="inner",
    )
    .merge(
        dx_summary,
        on=ID_COL,
        how="inner",
    )
    .merge(
        temporal_summary,
        on=ID_COL,
        how="left",
    )
)

print("\nTabular longitudinal dataset:", tabular.shape)
print(tabular.head())

# %%
# ============================================================
# 6. DEFINE TABULAR MODEL FEATURES
# ============================================================

NUMERIC_COLS = [
    "prior_visit_count",
    "prior_dx_count",
    "prior_unique_dx_count",
    "prior_icd9_count",
    "prior_icd10_count",
    "mean_codes_per_visit",
    "max_codes_per_visit",
    "history_span_days",
    "last_prior_visit_gap_days",
]

BINARY_COLS = [
    "prior_diabetes",
    "prior_ckd",
    "prior_hypertension",
    "prior_heart_failure",
    "prior_ischemic_heart_disease",
    "prior_cerebrovascular",
    "prior_obesity",
    "prior_dyslipidemia",
    "prior_diabetic_neuropathy",
]

# No trustworthy admission-level categorical feature is needed here because the
# old structured_features file refers to the old index admission.
CATEGORICAL_COLS = []

FEATURE_COLS = (
    NUMERIC_COLS
    + BINARY_COLS
)

for c in NUMERIC_COLS:
    tabular[c] = (
        pd.to_numeric(
            tabular[c],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

for c in BINARY_COLS:
    tabular[c] = (
        pd.to_numeric(
            tabular[c],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

train_tab = (
    tabular[
        tabular["split"] == "train"
    ]
    .reset_index(drop=True)
)

val_tab = (
    tabular[
        tabular["split"] == "validation"
    ]
    .reset_index(drop=True)
)

test_tab = (
    tabular[
        tabular["split"] == "test"
    ]
    .reset_index(drop=True)
)

print("\nTrain:", train_tab.shape)
print("Validation:", val_tab.shape)
print("Test:", test_tab.shape)

# %%
# ============================================================
# 7. COMMON METRICS
# ============================================================

def choose_threshold(y_true, y_prob):
    """
    Choose F1-maximising threshold using validation data only.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    scores = [
        f1_score(
            y_true,
            (y_prob >= t).astype(int),
            zero_division=0,
        )
        for t in thresholds
    ]

    return float(
        thresholds[
            int(np.argmax(scores))
        ]
    )


def binary_metrics(
    y_true,
    y_prob,
    threshold,
):
    y_true = np.asarray(
        y_true
    ).astype(int)

    y_prob = np.asarray(
        y_prob
    ).reshape(-1)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    return {
        "AUROC": roc_auc_score(
            y_true,
            y_prob,
        ),
        "AUPRC": average_precision_score(
            y_true,
            y_prob,
        ),
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall_Sensitivity": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Specificity": specificity,
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Brier": brier_score_loss(
            y_true,
            y_prob,
        ),
        "Threshold": threshold,
    }

# %%
# ============================================================
# 8. BUILD PATIENT SEQUENCES
# ============================================================

patient_sequence = defaultdict(list)

for row in dx_enriched[
    [
        ID_COL,
        "visit_number",
        "code_rank_within_visit",
        "icd_token",
    ]
].itertuples(index=False):

    patient_sequence[
        int(row.subject_id)
    ].append(
        (
            int(row.visit_number),
            int(row.code_rank_within_visit),
            str(row.icd_token),
        )
    )


def prior_codes_as_text(
    subject_id,
    max_codes=192,
):
    seq = patient_sequence.get(
        int(subject_id),
        [],
    )

    if not seq:
        return (
            "No prior diagnosis codes."
        )

    seq = seq[-max_codes:]

    # Explicit visit delimiters.
    pieces = []
    previous_visit = None

    for visit_no, _, token in seq:
        if visit_no != previous_visit:
            pieces.append(
                f"Visit {visit_no}:"
            )
            previous_visit = visit_no

        pieces.append(token)

    return " ".join(pieces)

# %%
# ============================================================
# 9. CLINICALBERT EHR-TO-TEXT DATA
# ============================================================
"""
No clinical-note file is supplied for this longitudinal cohort.
Therefore ClinicalBERT is used as an EHR-to-text adaptation:
- prior visit count
- history length
- selected pre-index comorbid diagnosis indicators
- prior ICD sequence

Report this as "ClinicalBERT EHR-to-text adaptation".
"""

clinicalbert_frame = tabular[
    [
        ID_COL,
        TARGET_COL,
        "split",
    ]
].copy()

summary_lookup = (
    tabular
    .set_index(ID_COL)
)


def make_clinicalbert_text(subject_id):
    row = summary_lookup.loc[
        subject_id
    ]

    present_conditions = []

    condition_names = {
        "prior_diabetes":
            "diabetes",
        "prior_ckd":
            "chronic kidney disease",
        "prior_hypertension":
            "hypertension",
        "prior_heart_failure":
            "heart failure",
        "prior_ischemic_heart_disease":
            "ischemic heart disease",
        "prior_cerebrovascular":
            "cerebrovascular disease",
        "prior_obesity":
            "obesity",
        "prior_dyslipidemia":
            "dyslipidemia",
        "prior_diabetic_neuropathy":
            "diabetic neuropathy",
    }

    for col, readable in condition_names.items():
        if int(row[col]) == 1:
            present_conditions.append(
                readable
            )

    condition_text = (
        ", ".join(present_conditions)
        if present_conditions
        else "none of the selected prior conditions"
    )

    text = (
        f"The patient has {int(row['prior_visit_count'])} "
        f"previous hospital visits and "
        f"{int(row['prior_dx_count'])} prior diagnosis records. "
        f"Prior documented conditions include {condition_text}. "
        f"Previous diagnosis sequence: "
        f"{prior_codes_as_text(subject_id)}"
    )

    return text


clinicalbert_frame[
    "clinical_text"
] = clinicalbert_frame[
    ID_COL
].apply(
    make_clinicalbert_text
)

print("\nExample ClinicalBERT input:")
print(
    clinicalbert_frame[
        "clinical_text"
    ].iloc[0][:1200]
)

# %%
# SMALL CLINICALBERT TEST DATASET

train_small = clinicalbert_frame[
    clinicalbert_frame["split"] == "train"
].sample(
    n=300,
    random_state=42
)

val_small = clinicalbert_frame[
    clinicalbert_frame["split"] == "validation"
].sample(
    n=100,
    random_state=42
)

test_small = clinicalbert_frame[
    clinicalbert_frame["split"] == "test"
].sample(
    n=100,
    random_state=42
)

clinicalbert_small = pd.concat([
    train_small,
    val_small,
    test_small
]).reset_index(drop=True)

print(clinicalbert_small["split"].value_counts())
print("\nTotal patients:", len(clinicalbert_small))

# %%
# ============================================================
# 10. CLINICALBERT MODEL
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

CLINICALBERT_NAME = (
    "./Bio_ClinicalBERT"
)


class ClinicalBERTDataset(Dataset):
    def __init__(
        self,
        frame,
        tokenizer,
        max_len=256,
    ):
        self.texts = (
            frame["clinical_text"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            frame[TARGET_COL]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            k: v.squeeze(0)
            for k, v in enc.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


@torch.no_grad()
def predict_clinicalbert(
    model,
    loader,
):
    model.eval()

    y_all = []
    p_all = []

    for batch in loader:
        y = batch.pop(
            "labels"
        ).to(DEVICE)

        x = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        logits = model(
            **x
        ).logits

        prob = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        y_all.extend(
            y.cpu().numpy()
        )

        p_all.extend(
            prob.cpu().numpy()
        )

    return (
        np.asarray(y_all),
        np.asarray(p_all),
    )


def train_clinicalbert(
    frame,
    epochs=3,
    batch_size=None,
    lr=2e-5,
    max_len=CLINICALBERT_MAX_LEN,
    patience=2,
):
    if batch_size is None:
        batch_size = (
            8
            if DEVICE.type == "cuda"
            else 2
        )

    tr = frame[
        frame["split"] == "train"
    ].reset_index(drop=True)

    va = frame[
        frame["split"] == "validation"
    ].reset_index(drop=True)

    te = frame[
        frame["split"] == "test"
    ].reset_index(drop=True)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            "./Bio_ClinicalBERT", local_files_only=True
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            "./Bio_ClinicalBERT", local_files_only=True,
            num_labels=2,
        )
        .to(DEVICE)
    )

    tr_loader = DataLoader(
        ClinicalBERTDataset(
            tr,
            tokenizer,
            max_len,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    va_loader = DataLoader(
        ClinicalBERTDataset(
            va,
            tokenizer,
            max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    te_loader = DataLoader(
        ClinicalBERTDataset(
            te,
            tokenizer,
            max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    y_train = (
        tr[TARGET_COL]
        .to_numpy()
    )

    counts = np.bincount(
        y_train,
        minlength=2,
    )

    weights = (
        len(y_train)
        / (
            2
            * np.maximum(
                counts,
                1,
            )
        )
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=weights
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    best_auc = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        running_loss = 0.0

        for batch in tr_loader:
            labels_batch = (
                batch.pop(
                    "labels"
                ).to(DEVICE)
            )

            x = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            logits = model(
                **x
            ).logits

            loss = criterion(
                logits,
                labels_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

        y_val, p_val = (
            predict_clinicalbert(
                model,
                va_loader,
            )
        )

        val_auc = roc_auc_score(
            y_val,
            p_val,
        )

        print(
            f"ClinicalBERT epoch {epoch}: "
            f"loss="
            f"{running_loss/max(len(tr_loader),1):.4f} "
            f"val_AUROC={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )
            no_improve = 0

        else:
            no_improve += 1

            if (
                no_improve
                >= patience
            ):
                print(
                    "ClinicalBERT "
                    "early stopping."
                )
                break

    model.load_state_dict(
        best_state
    )

    y_val, p_val = (
        predict_clinicalbert(
            model,
            va_loader,
        )
    )

    threshold = choose_threshold(
        y_val,
        p_val,
    )

    y_test, p_test = (
        predict_clinicalbert(
            model,
            te_loader,
        )
    )

    metrics = binary_metrics(
        y_test,
        p_test,
        threshold,
    )

    out = (
        DATA_DIR
        / "checkpoints"
        / "clinicalbert_longitudinal_dn"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    return (
        model,
        metrics,
        p_test,
    )

# %%
# ============================================================
# 11. MED-BERT-STYLE MODEL
# ============================================================

PAD = "[PAD]"
UNK = "[UNK]"


def build_medbert_vocab(
    dx_frame,
    train_subject_ids,
    min_freq=1,
):
    train_ids = set(
        int(x)
        for x in train_subject_ids
    )

    counts = Counter(
        dx_frame.loc[
            dx_frame[ID_COL].isin(
                train_ids
            ),
            "icd_token",
        ]
    )

    vocab = {
        PAD: 0,
        UNK: 1,
    }

    for token, n in sorted(
        counts.items()
    ):
        if n >= min_freq:
            vocab[token] = (
                len(vocab)
            )

    return vocab


train_ids = (
    model_cohort.loc[
        model_cohort[
            "split"
        ] == "train",
        ID_COL,
    ]
    .tolist()
)

MEDBERT_VOCAB = (
    build_medbert_vocab(
        dx_enriched,
        train_ids,
    )
)

print(
    "Med-BERT-style "
    "vocabulary size:",
    len(MEDBERT_VOCAB),
)


class MedBERTDataset(Dataset):
    def __init__(
        self,
        patients,
        sequence_dict,
        vocab,
        max_len=256,
        max_visits=256,
        max_rank=128,
    ):
        self.patients = (
            patients
            .reset_index(drop=True)
        )

        self.sequence_dict = (
            sequence_dict
        )

        self.vocab = vocab
        self.max_len = max_len
        self.max_visits = max_visits
        self.max_rank = max_rank

    def __len__(self):
        return len(
            self.patients
        )

    def __getitem__(
        self,
        idx,
    ):
        row = (
            self.patients
            .iloc[idx]
        )

        sid = int(
            row[ID_COL]
        )

        seq = (
            self.sequence_dict
            .get(
                sid,
                [],
            )
        )

        # Keep most recent codes.
        seq = seq[
            -self.max_len:
        ]

        if len(seq) == 0:
            token_ids = [
                self.vocab[UNK]
            ]
            visit_ids = [0]
            rank_ids = [0]

        else:
            visits = sorted(
                set(
                    visit_no
                    for (
                        visit_no,
                        _,
                        _,
                    ) in seq
                )
            )

            visit_map = {
                visit_no: i
                for i, visit_no
                in enumerate(visits)
            }

            token_ids = []
            visit_ids = []
            rank_ids = []

            for (
                visit_no,
                rank,
                token,
            ) in seq:

                token_ids.append(
                    self.vocab.get(
                        token,
                        self.vocab[UNK],
                    )
                )

                visit_ids.append(
                    min(
                        visit_map[
                            visit_no
                        ],
                        self.max_visits
                        - 1,
                    )
                )

                rank_ids.append(
                    min(
                        max(
                            rank - 1,
                            0,
                        ),
                        self.max_rank
                        - 1,
                    )
                )

        length = len(
            token_ids
        )

        pad_len = (
            self.max_len
            - length
        )

        token_ids += (
            [0] * pad_len
        )

        visit_ids += (
            [0] * pad_len
        )

        rank_ids += (
            [0] * pad_len
        )

        attention_mask = (
            [1] * length
            + [0] * pad_len
        )

        return {
            "input_ids":
                torch.tensor(
                    token_ids,
                    dtype=torch.long,
                ),
            "visit_ids":
                torch.tensor(
                    visit_ids,
                    dtype=torch.long,
                ),
            "rank_ids":
                torch.tensor(
                    rank_ids,
                    dtype=torch.long,
                ),
            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.bool,
                ),
            "labels":
                torch.tensor(
                    int(
                        row[
                            TARGET_COL
                        ]
                    ),
                    dtype=torch.long,
                ),
        }


class MedBERTStyleClassifier(
    nn.Module
):
    def __init__(
        self,
        vocab_size,
        hidden_dim=192,
        n_heads=6,
        n_layers=6,
        ff_dim=768,
        dropout=0.1,
        max_visits=256,
        max_rank=128,
    ):
        super().__init__()

        self.code_embedding = (
            nn.Embedding(
                vocab_size,
                hidden_dim,
                padding_idx=0,
            )
        )

        self.visit_embedding = (
            nn.Embedding(
                max_visits,
                hidden_dim,
            )
        )

        self.rank_embedding = (
            nn.Embedding(
                max_rank,
                hidden_dim,
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=n_layers,
            )
        )

        self.norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.classifier = (
            nn.Linear(
                hidden_dim,
                2,
            )
        )

    def forward(
        self,
        input_ids,
        visit_ids,
        rank_ids,
        attention_mask,
    ):
        x = (
            self.code_embedding(
                input_ids
            )
            + self.visit_embedding(
                visit_ids
            )
            + self.rank_embedding(
                rank_ids
            )
        )

        h = self.encoder(
            x,
            src_key_padding_mask=(
                ~attention_mask
            ),
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            (h * mask)
            .sum(dim=1)
            / mask.sum(dim=1)
            .clamp(min=1.0)
        )

        pooled = self.norm(
            pooled
        )

        return self.classifier(
            self.dropout(
                pooled
            )
        )


@torch.no_grad()
def predict_medbert(
    model,
    loader,
):
    model.eval()

    y_all = []
    p_all = []

    for batch in loader:
        y = (
            batch.pop(
                "labels"
            ).to(DEVICE)
        )

        x = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        logits = model(
            **x
        )

        prob = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        )

        y_all.extend(
            y.cpu().numpy()
        )

        p_all.extend(
            prob.cpu().numpy()
        )

    return (
        np.asarray(y_all),
        np.asarray(p_all),
    )


def train_medbert_style(
    frame,
    epochs=20,
    batch_size=None,
    lr=1e-4,
    max_len=MEDBERT_MAX_LEN,
    patience=4,
):
    if batch_size is None:
        batch_size = (
            32
            if DEVICE.type == "cuda"
            else 8
        )

    tr = frame[
        frame["split"] == "train"
    ].reset_index(drop=True)

    va = frame[
        frame["split"] == "validation"
    ].reset_index(drop=True)

    te = frame[
        frame["split"] == "test"
    ].reset_index(drop=True)

    tr_loader = DataLoader(
        MedBERTDataset(
            tr,
            patient_sequence,
            MEDBERT_VOCAB,
            max_len=max_len,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    va_loader = DataLoader(
        MedBERTDataset(
            va,
            patient_sequence,
            MEDBERT_VOCAB,
            max_len=max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    te_loader = DataLoader(
        MedBERTDataset(
            te,
            patient_sequence,
            MEDBERT_VOCAB,
            max_len=max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = (
        MedBERTStyleClassifier(
            vocab_size=len(
                MEDBERT_VOCAB
            )
        )
        .to(DEVICE)
    )

    y_train = (
        tr[TARGET_COL]
        .to_numpy()
    )

    counts = np.bincount(
        y_train,
        minlength=2,
    )

    weights = (
        len(y_train)
        / (
            2
            * np.maximum(
                counts,
                1,
            )
        )
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=weights
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    best_auc = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        running_loss = 0.0

        for batch in tr_loader:
            labels_batch = (
                batch.pop(
                    "labels"
                ).to(DEVICE)
            )

            x = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            logits = model(
                **x
            )

            loss = criterion(
                logits,
                labels_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

        y_val, p_val = (
            predict_medbert(
                model,
                va_loader,
            )
        )

        val_auc = roc_auc_score(
            y_val,
            p_val,
        )

        print(
            f"Med-BERT-style epoch {epoch}: "
            f"loss="
            f"{running_loss/max(len(tr_loader),1):.4f} "
            f"val_AUROC={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )
            no_improve = 0

        else:
            no_improve += 1

            if no_improve >= patience:
                print(
                    "Med-BERT-style "
                    "early stopping."
                )
                break

    model.load_state_dict(
        best_state
    )

    y_val, p_val = (
        predict_medbert(
            model,
            va_loader,
        )
    )

    threshold = choose_threshold(
        y_val,
        p_val,
    )

    y_test, p_test = (
        predict_medbert(
            model,
            te_loader,
        )
    )

    metrics = binary_metrics(
        y_test,
        p_test,
        threshold,
    )

    out = (
        DATA_DIR
        / "checkpoints"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "vocab":
                MEDBERT_VOCAB,
            "target":
                TARGET_COL,
            "max_len":
                max_len,
        },
        out
        / "medbert_style_longitudinal_dn.pt",
    )

    return (
        model,
        metrics,
        p_test,
    )

# %%
# ============================================================
# 12. TRANSTAB
# ============================================================

def train_transtab(
    train_frame,
    val_frame,
    test_frame,
    epochs=50,
    batch_size=None,
    lr=1e-4,
):
    import transtab

    if batch_size is None:
        batch_size = (
            128
            if DEVICE.type == "cuda"
            else 32
        )

    transtab.random_seed(
        SEED
    )

    X_train = train_frame[
        FEATURE_COLS
    ].copy()

    y_train = train_frame[
        TARGET_COL
    ].astype(int).copy()

    X_val = val_frame[
        FEATURE_COLS
    ].copy()

    y_val = val_frame[
        TARGET_COL
    ].astype(int).copy()

    X_test = test_frame[
        FEATURE_COLS
    ].copy()

    y_test = test_frame[
        TARGET_COL
    ].astype(int).to_numpy()

    model = (
        transtab
        .build_classifier(
            CATEGORICAL_COLS,
            NUMERIC_COLS,
            BINARY_COLS,
        )
    )

    out = (
        DATA_DIR
        / "checkpoints"
        / "transtab_longitudinal_dn"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_arguments = {
        "num_epoch":
            epochs,
        "batch_size":
            batch_size,
        "lr":
            lr,
        "eval_metric":
            "val_loss",
        "eval_less_is_better":
            True,
        "output_dir":
            str(out),
    }

    transtab.train(
        model,
        (X_train, y_train),
        (X_val, y_val),
        **training_arguments,
    )

    p_val = np.asarray(
        transtab.predict(
            model,
            X_val,
        )
    ).reshape(-1)

    if (
        np.nanmin(p_val) < 0
        or np.nanmax(p_val) > 1
    ):
        p_val = (
            1
            / (
                1
                + np.exp(
                    -p_val
                )
            )
        )

    threshold = choose_threshold(
        y_val.to_numpy(),
        p_val,
    )

    p_test = np.asarray(
        transtab.predict(
            model,
            X_test,
        )
    ).reshape(-1)

    if (
        np.nanmin(p_test) < 0
        or np.nanmax(p_test) > 1
    ):
        p_test = (
            1
            / (
                1
                + np.exp(
                    -p_test
                )
            )
        )

    metrics = binary_metrics(
        y_test,
        p_test,
        threshold,
    )

    return (
        model,
        metrics,
        p_test,
    )

# %%
# ============================================================
# 13. FT-TRANSFORMER
# ============================================================

def find_positive_probability_column(
    prediction_df,
):
    candidates = [
        f"{TARGET_COL}_1_probability",
        "1_probability",
    ]

    for col in candidates:
        if col in prediction_df.columns:
            return col

    probability_cols = [
        c
        for c in prediction_df.columns
        if "probability"
        in str(c).lower()
    ]

    class1 = [
        c
        for c in probability_cols
        if (
            str(c)
            .lower()
            .endswith(
                "1_probability"
            )
            or "_1_"
            in str(c).lower()
        )
    ]

    if len(class1) == 1:
        return class1[0]

    raise ValueError(
        "Could not identify the "
        "positive-class probability column. "
        f"Columns: "
        f"{prediction_df.columns.tolist()}"
    )


def train_fttransformer(
    train_frame,
    val_frame,
    test_frame,
    epochs=50,
    batch_size=None,
    lr=1e-3,
):
    from pytorch_tabular import (
        TabularModel,
    )

    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
    )

    from pytorch_tabular.models import (
        FTTransformerConfig,
    )

    if batch_size is None:
        batch_size = (
            256
            if DEVICE.type == "cuda"
            else 64
        )

    # Binary flags are treated as categorical variables.
    ft_cat_cols = (
        BINARY_COLS.copy()
    )

    keep_cols = (
        NUMERIC_COLS
        + ft_cat_cols
        + [TARGET_COL]
    )

    tr = train_frame[
        keep_cols
    ].copy()

    va = val_frame[
        keep_cols
    ].copy()

    te = test_frame[
        keep_cols
    ].copy()

    # Convert binary categorical variables to strings for robust category handling.
    for c in ft_cat_cols:
        tr[c] = tr[c].astype(str)
        va[c] = va[c].astype(str)
        te[c] = te[c].astype(str)

    data_config = DataConfig(
        target=[TARGET_COL],
        continuous_cols=NUMERIC_COLS,
        categorical_cols=ft_cat_cols,
        normalize_continuous_features=True,
    )

    out = (
        DATA_DIR
        / "checkpoints"
        / "fttransformer_longitudinal_dn"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=epochs,
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=5,
        checkpoints="valid_loss",
        checkpoints_path=str(out),
        load_best=True,
        accelerator="auto",
    )

    optimizer_config = (
        OptimizerConfig()
    )

    model_config = (
        FTTransformerConfig(
            task="classification",
            input_embed_dim=32,
            num_attn_blocks=3,
            num_heads=4,
            learning_rate=lr,
        )
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True,
    )

    tabular_model.fit(
        train=tr,
        validation=va,
    )

    val_pred = (
        tabular_model.predict(
            va,
            include_input_features=False,
        )
    )

    test_pred = (
        tabular_model.predict(
            te,
            include_input_features=False,
        )
    )

    prob_col = (
        find_positive_probability_column(
            val_pred
        )
    )

    p_val = (
        val_pred[
            prob_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    threshold = choose_threshold(
        va[TARGET_COL].to_numpy(),
        p_val,
    )

    p_test = (
        test_pred[
            prob_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    metrics = binary_metrics(
        te[TARGET_COL].to_numpy(),
        p_test,
        threshold,
    )

    return (
        tabular_model,
        metrics,
        p_test,
    )

# %%
# ============================================================
# 14. TRAIN FOUR MODELS
# ============================================================

results = []

# Test patient order is fixed here.
test_reference = (
    model_cohort[
        model_cohort[
            "split"
        ] == "test"
    ][
        [
            ID_COL,
            TARGET_COL,
        ]
    ]
    .reset_index(drop=True)
)

predictions = (
    test_reference
    .rename(
        columns={
            TARGET_COL:
                "y_true"
        }
    )
)

# ClinicalBERT data order needs to follow split-filter order.
clinicalbert_test_ids = (
    clinicalbert_small[
        clinicalbert_small["split"] == "test"
    ][ID_COL]
    .reset_index(drop=True)
)

# Med-BERT data order follows model_cohort test order.
medbert_test_ids = (
    model_cohort[
        model_cohort[
            "split"
        ] == "test"
    ][ID_COL]
    .reset_index(drop=True)
)


if RUN_CLINICALBERT:
    print(
        "\n"
        + "=" * 70
    )
    print(
        "TRAINING CLINICALBERT "
        "EHR-TO-TEXT"
    )
    print(
        "=" * 70
    )

    (
    clinicalbert_model,
    clinicalbert_metrics,
    p_clinicalbert,
) = train_clinicalbert(
    clinicalbert_small,
    epochs=1,
    batch_size=2,
    max_len=128
)

    results.append(
        {
            "Model":
                "ClinicalBERT EHR-to-text",
            **clinicalbert_metrics,
        }
    )

    clinical_pred = pd.DataFrame(
        {
            ID_COL:
                clinicalbert_test_ids,
            "ClinicalBERT_probability":
                p_clinicalbert,
        }
    )

    predictions = (
        predictions
        .merge(
            clinical_pred,
            on=ID_COL,
            how="left",
        )
    )


if RUN_MEDBERT:
    print(
        "\n"
        + "=" * 70
    )
    print(
        "TRAINING MED-BERT-STYLE"
    )
    print(
        "=" * 70
    )

    (
        medbert_model,
        medbert_metrics,
        p_medbert,
    ) = train_medbert_style(
        model_cohort
    )

    results.append(
        {
            "Model":
                "Med-BERT-style",
            **medbert_metrics,
        }
    )

    medbert_pred = pd.DataFrame(
        {
            ID_COL:
                medbert_test_ids,
            "MedBERT_style_probability":
                p_medbert,
        }
    )

    predictions = (
        predictions
        .merge(
            medbert_pred,
            on=ID_COL,
            how="left",
        )
    )


if RUN_TRANSTAB:
    print(
        "\n"
        + "=" * 70
    )
    print(
        "TRAINING TRANSTAB"
    )
    print(
        "=" * 70
    )

    (
        transtab_model,
        transtab_metrics,
        p_transtab,
    ) = train_transtab(
        train_tab,
        val_tab,
        test_tab,
    )

    results.append(
        {
            "Model":
                "TransTab",
            **transtab_metrics,
        }
    )

    transtab_pred = pd.DataFrame(
        {
            ID_COL:
                test_tab[
                    ID_COL
                ].to_numpy(),
            "TransTab_probability":
                p_transtab,
        }
    )

    predictions = (
        predictions
        .merge(
            transtab_pred,
            on=ID_COL,
            how="left",
        )
    )


if RUN_FTTRANSFORMER:
    print(
        "\n"
        + "=" * 70
    )
    print(
        "TRAINING FT-TRANSFORMER"
    )
    print(
        "=" * 70
    )

    (
        ft_model,
        ft_metrics,
        p_ft,
    ) = train_fttransformer(
        train_tab,
        val_tab,
        test_tab,
    )

    results.append(
        {
            "Model":
                "FT-Transformer",
            **ft_metrics,
        }
    )

    ft_pred = pd.DataFrame(
        {
            ID_COL:
                test_tab[
                    ID_COL
                ].to_numpy(),
            "FTTransformer_probability":
                p_ft,
        }
    )

    predictions = (
        predictions
        .merge(
            ft_pred,
            on=ID_COL,
            how="left",
        )
    )

# %%
# ============================================================
# 15. SAVE FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

if not results_df.empty:

    results_df = (
        results_df
        .sort_values(
            "AUROC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\nFINAL MODEL COMPARISON"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    results_file = (
        DATA_DIR
        / "dn_longitudinal_four_model_comparison.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

predictions_file = (
    DATA_DIR
    / "dn_longitudinal_test_predictions.csv"
)

predictions.to_csv(
    predictions_file,
    index=False,
)

feature_file = (
    DATA_DIR
    / "dn_longitudinal_tabular_features.parquet"
)

tabular.to_parquet(
    feature_file,
    index=False,
)

config_file = (
    DATA_DIR
    / "dn_longitudinal_experiment_config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "target":
                TARGET_COL,
            "common_patients":
                int(
                    model_cohort[
                        ID_COL
                    ].nunique()
                ),
            "numeric_features":
                NUMERIC_COLS,
            "binary_features":
                BINARY_COLS,
            "categorical_features":
                CATEGORICAL_COLS,
            "seed":
                SEED,
            "device":
                str(DEVICE),
            "clinicalbert_max_len":
                CLINICALBERT_MAX_LEN,
            "medbert_max_len":
                MEDBERT_MAX_LEN,
        },
        f,
        indent=2,
    )

print("\nSaved outputs:")

if not results_df.empty:
    print(results_file)

print(predictions_file)
print(feature_file)
print(config_file)

# %%
# ============================================================
# 16. IMPORTANT FOR THE NEXT STAGE
# ============================================================
"""
If you want to include:
- first-24h creatinine
- BUN
- glucose
- SBP
- admission demographics
- medications

in TransTab / FT-Transformer / ClinicalBERT, regenerate those feature files
using cohort_labels_longitudinal.parquet and its NEW index_hadm_id.

Do NOT merge the old early_features_24h.parquet, early_features_48h.parquet,
or structured_features.parquet into this cohort without checking that their
index_hadm_id matches the new longitudinal index admission.
"""


