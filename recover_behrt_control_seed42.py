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
    brier_score_loss,
)

# ============================================================
# SETTINGS — EXACT CONTROL CONFIGURATION
# ============================================================

MAX_LEN = 256
MAX_VOCAB = 14000

HIDDEN = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 32

CHECKPOINT = "behrt_365d_control_best.pt"

OUT_PRED = (
    "behrt_365d_control_seed42_recovered_predictions.csv"
)

OUT_RESULT = (
    "behrt_365d_control_seed42_recovered_results.csv"
)

# CPU is sufficient because this is inference only.
DEVICE = torch.device("cpu")

torch.set_num_threads(4)

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)


# ============================================================
# LOAD EXACT LEAKAGE-SAFE DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_behrt_365d.parquet"
)

cohort = pd.read_parquet(
    "cohort_labels_timeaware_365d.parquet"
)[
    ["subject_id", "dn_within_365d"]
]

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

cohort = cohort.merge(
    splits,
    on="subject_id",
    how="inner"
)

seq["subject_id"] = (
    seq["subject_id"]
    .astype(int)
)

cohort["subject_id"] = (
    cohort["subject_id"]
    .astype(int)
)

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

print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# TRAINING-ONLY VOCABULARY
# ============================================================

train_codes = seq[
    seq["subject_id"].isin(train_ids)
]["icd_token"].value_counts()

top_codes = train_codes.index[
    : MAX_VOCAB - 2
].tolist()

code_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for i, code in enumerate(
    top_codes,
    start=2
):
    code_to_id[code] = i

VOCAB_SIZE = len(code_to_id)

print("Vocabulary size:", VOCAB_SIZE)


# ============================================================
# LABEL LOOKUP
# ============================================================

label_map = dict(
    zip(
        cohort["subject_id"],
        cohort["dn_within_365d"]
    )
)


# ============================================================
# BUILD IDENTICAL PATIENT SEQUENCES
# ============================================================

seq = seq.sort_values(
    [
        "subject_id",
        "admittime",
        "visit_number",
        "code_rank"
    ]
)

patient_data = {}
truncated = 0

for sid, g in seq.groupby(
    "subject_id"
):

    g = g.sort_values(
        [
            "admittime",
            "visit_number",
            "code_rank"
        ]
    )

    if len(g) > MAX_LEN:
        truncated += 1
        g = g.iloc[-MAX_LEN:]

    codes = [
        code_to_id.get(x, 1)
        for x in g["icd_token"]
    ]

    ages = (
        g["visit_age"]
        .fillna(0)
        .astype(int)
        .clip(0, 120)
        .tolist()
    )

    segments = (
        (
            (
                g["visit_number"]
                .astype(int)
                - 1
            ) % 2
        )
        .tolist()
    )

    gaps = (
        g["time_gap_bucket"]
        .astype(int)
        .clip(0, 8)
        .tolist()
    )

    recency = (
        g["time_to_cutoff_bucket"]
        .astype(int)
        .clip(0, 8)
        .tolist()
    )

    patient_data[int(sid)] = {
        "codes": codes,
        "ages": ages,
        "segments": segments,
        "gaps": gaps,
        "recency": recency
    }

print(
    "Patients truncated:",
    truncated
)


# ============================================================
# DATASET
# ============================================================

class BEHRTDataset(Dataset):

    def __init__(
        self,
        patient_ids
    ):
        self.patient_ids = patient_ids

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

        codes = p["codes"]
        ages = p["ages"]
        segments = p["segments"]
        gaps = p["gaps"]
        recency = p["recency"]

        length = len(codes)

        positions = list(
            range(length)
        )

        pad = MAX_LEN - length

        codes = (
            codes
            + [0] * pad
        )

        ages = (
            ages
            + [0] * pad
        )

        segments = (
            segments
            + [0] * pad
        )

        gaps = (
            gaps
            + [0] * pad
        )

        recency = (
            recency
            + [0] * pad
        )

        positions = (
            positions
            + [0] * pad
        )

        return {
            "subject_id": sid,

            "codes": torch.tensor(
                codes,
                dtype=torch.long
            ),

            "ages": torch.tensor(
                ages,
                dtype=torch.long
            ),

            "segments": torch.tensor(
                segments,
                dtype=torch.long
            ),

            "positions": torch.tensor(
                positions,
                dtype=torch.long
            ),

            "gaps": torch.tensor(
                gaps,
                dtype=torch.long
            ),

            "recency": torch.tensor(
                recency,
                dtype=torch.long
            ),

            "label": torch.tensor(
                float(label_map[sid]),
                dtype=torch.float32
            )
        }


val_loader = DataLoader(
    BEHRTDataset(val_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    BEHRTDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# EXACT BEHRT CONTROL ARCHITECTURE
# ============================================================

class TimeAwareBEHRT(nn.Module):

    def __init__(self):

        super().__init__()

        self.code_embedding = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN,
            padding_idx=0
        )

        self.age_embedding = nn.Embedding(
            121,
            HIDDEN
        )

        self.segment_embedding = nn.Embedding(
            2,
            HIDDEN
        )

        self.position_embedding = nn.Embedding(
            MAX_LEN,
            HIDDEN
        )

        # Present in checkpoint architecture
        # but deliberately NOT used by control forward pass.
        self.time_gap_embedding = (
            nn.Embedding(
                9,
                HIDDEN
            )
        )

        self.time_to_cutoff_embedding = (
            nn.Embedding(
                9,
                HIDDEN
            )
        )

        self.layer_norm = (
            nn.LayerNorm(
                HIDDEN
            )
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=HIDDEN,
                nhead=N_HEADS,
                dim_feedforward=HIDDEN * 4,
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
        segments,
        positions,
        gaps,
        recency
    ):

        padding_mask = (
            codes.eq(0)
        )

        # IMPORTANT:
        # Exact temporal-ablation control.
        # gaps and recency are NOT added.
        x = (
            self.code_embedding(
                codes
            )
            + self.age_embedding(
                ages
            )
            + self.segment_embedding(
                segments
            )
            + self.position_embedding(
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


# ============================================================
# LOAD SURVIVING CONTROL CHECKPOINT
# ============================================================

model = TimeAwareBEHRT().to(
    DEVICE
)

state = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    state,
    strict=True
)

model.eval()

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# PREDICTION
# ============================================================

def predict(loader):

    subject_ids = []
    labels = []
    probabilities = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            codes = (
                batch["codes"]
                .to(DEVICE)
            )

            ages = (
                batch["ages"]
                .to(DEVICE)
            )

            segments = (
                batch["segments"]
                .to(DEVICE)
            )

            positions = (
                batch["positions"]
                .to(DEVICE)
            )

            gaps = (
                batch["gaps"]
                .to(DEVICE)
            )

            recency = (
                batch["recency"]
                .to(DEVICE)
            )

            logits = model(
                codes,
                ages,
                segments,
                positions,
                gaps,
                recency
            )

            probs = torch.sigmoid(
                logits
            )

            subject_ids.extend(
                batch["subject_id"]
                .cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch["label"]
                .cpu()
                .numpy()
                .tolist()
            )

            probabilities.extend(
                probs
                .cpu()
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
# RECOVER VALIDATION THRESHOLD
# ============================================================

_, val_y, val_prob = predict(
    val_loader
)

best_threshold = 0.50
best_f1 = -1.0

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
        best_threshold = float(
            threshold
        )

print(
    "Recovered validation threshold:",
    round(best_threshold, 3)
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_subjects, test_y, test_prob = (
    predict(test_loader)
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
    )
    .ravel()
)

auc = roc_auc_score(
    test_y,
    test_prob
)

auprc = average_precision_score(
    test_y,
    test_prob
)

accuracy = accuracy_score(
    test_y,
    test_pred
)

precision = precision_score(
    test_y,
    test_pred,
    zero_division=0
)

sensitivity = recall_score(
    test_y,
    test_pred,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
)

f1 = f1_score(
    test_y,
    test_pred,
    zero_division=0
)

brier = brier_score_loss(
    test_y,
    test_prob
)


# ============================================================
# SAVE TO NEW FILES ONLY
# ============================================================

pd.DataFrame({
    "subject_id": test_subjects,
    "y_true": test_y,
    "y_prob": test_prob,
    "y_pred": test_pred
}).to_csv(
    OUT_PRED,
    index=False
)

pd.DataFrame([{
    "Model":
        "BEHRT Control 365d - Seed 42 Recovered",

    "N_test":
        len(test_y),

    "AUROC":
        auc,

    "AUPRC":
        auprc,

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall_Sensitivity":
        sensitivity,

    "Specificity":
        specificity,

    "F1":
        f1,

    "Brier":
        brier,

    "Threshold":
        best_threshold
}]).to_csv(
    OUT_RESULT,
    index=False
)


# ============================================================
# VERIFY AGAINST KNOWN ORIGINAL RESULT
# ============================================================

EXPECTED_AUROC = 0.689887
EXPECTED_AUPRC = 0.142548

auc_diff = abs(
    auc - EXPECTED_AUROC
)

auprc_diff = abs(
    auprc - EXPECTED_AUPRC
)

match = (
    auc_diff < 0.001
    and
    auprc_diff < 0.001
)

print()
print(
    "=" * 70
)
print(
    "RECOVERED BEHRT CONTROL SEED-42"
)
print(
    "=" * 70
)

print(
    f"AUROC: {auc:.6f}"
)

print(
    f"AUPRC: {auprc:.6f}"
)

print(
    f"Accuracy: {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Sensitivity: {sensitivity:.6f}"
)

print(
    f"Specificity: {specificity:.6f}"
)

print(
    f"F1: {f1:.6f}"
)

print(
    f"Brier: {brier:.6f}"
)

print(
    f"Threshold: {best_threshold:.3f}"
)

print()
print(
    "Expected original AUROC:",
    EXPECTED_AUROC
)

print(
    "Expected original AUPRC:",
    EXPECTED_AUPRC
)

print()
print(
    "RECOVERY MATCH:",
    "YES" if match else "NO"
)

print()
print(
    "Saved:",
    OUT_PRED
)

print(
    "Saved:",
    OUT_RESULT
)
