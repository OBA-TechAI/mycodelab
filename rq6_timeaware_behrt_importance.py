
"""RQ6 BEHRT Control: GPU-based feature masking importance."""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_LEN = 256
MAX_VOCAB = 14000
HIDDEN = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1
BATCH_SIZE = 32
TOP_N = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT = Path("rq6_timeaware_behrt_outputs")
OUT.mkdir(exist_ok=True)

print("Device:", DEVICE, flush=True)

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required: submit through sbatch.")


# ============================================================
# 2. LOAD ORIGINAL DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_behrt_365d.parquet"
)

cohort = pd.read_parquet(
    "cohort_labels_timeaware_365d.parquet"
)[["subject_id", "dn_within_365d"]]

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

cohort = cohort.merge(
    splits, on="subject_id", how="inner"
)

seq["subject_id"] = seq["subject_id"].astype(int)
cohort["subject_id"] = cohort["subject_id"].astype(int)

train_ids = cohort.loc[
    cohort["split"] == "train", "subject_id"
].tolist()

test_ids = cohort.loc[
    cohort["split"] == "test", "subject_id"
].tolist()

train_codes = seq[
    seq["subject_id"].isin(train_ids)
]["icd_token"].value_counts()

top_codes = train_codes.index[:MAX_VOCAB - 2].tolist()

code_to_id = {"<PAD>": 0, "<UNK>": 1}

for i, code in enumerate(top_codes, start=2):
    code_to_id[code] = i

VOCAB_SIZE = len(code_to_id)

label_map = dict(
    zip(cohort["subject_id"], cohort["dn_within_365d"])
)

seq = seq.sort_values([
    "subject_id", "admittime",
    "visit_number", "code_rank"
])

patient_data = {}

for sid, g in seq.groupby("subject_id"):

    g = g.sort_values([
        "admittime", "visit_number", "code_rank"
    ])

    if len(g) > MAX_LEN:
        g = g.iloc[-MAX_LEN:]

    patient_data[int(sid)] = {
        "codes": [
            code_to_id.get(c, 1)
            for c in g["icd_token"]
        ],
        "ages": (
            g["visit_age"].fillna(0)
            .astype(int).clip(0, 120).tolist()
        ),
        "segments": (
            (g["visit_number"].astype(int) - 1) % 2
        ).tolist(),
        "gaps": (
            g["time_gap_bucket"]
            .astype(int).clip(0, 8).tolist()
        ),
        "recency": (
            g["time_to_cutoff_bucket"]
            .astype(int).clip(0, 8).tolist()
        )
    }

print(
    "Vocabulary:", VOCAB_SIZE,
    "Test patients:", len(test_ids),
    flush=True
)


# ============================================================
# 3. DATASET
# ============================================================

class BEHRTDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        sid = int(self.ids[index])
        p = patient_data[sid]

        length = len(p["codes"])
        pad = MAX_LEN - length

        result = {"subject_id": sid}

        for key in [
            "codes", "ages", "segments",
            "gaps", "recency"
        ]:
            result[key] = torch.tensor(
                p[key] + [0] * pad,
                dtype=torch.long
            )

        result["positions"] = torch.tensor(
            list(range(length)) + [0] * pad,
            dtype=torch.long
        )

        result["label"] = int(label_map[sid])

        return result


# ============================================================
# 4. ORIGINAL BEHRT CONTROL ARCHITECTURE
# ============================================================

class TimeAwareBEHRT(nn.Module):

    def __init__(self):

        super().__init__()

        self.code_embedding = nn.Embedding(
            VOCAB_SIZE, HIDDEN, padding_idx=0
        )
        self.age_embedding = nn.Embedding(121, HIDDEN)
        self.segment_embedding = nn.Embedding(2, HIDDEN)
        self.position_embedding = nn.Embedding(MAX_LEN, HIDDEN)

        # Retained to match the original checkpoint.
        # These layers are NOT used by the control forward pass.
        self.time_gap_embedding = nn.Embedding(9, HIDDEN)
        self.time_to_cutoff_embedding = nn.Embedding(9, HIDDEN)

        self.layer_norm = nn.LayerNorm(HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS
        )

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1)
        )

    def forward(
        self, codes, ages, segments,
        positions, gaps, recency
    ):

        padding_mask = codes.eq(0)

        x = (
            self.code_embedding(codes)
            + self.age_embedding(ages)
            + self.segment_embedding(segments)
            + self.position_embedding(positions)
            + self.time_gap_embedding(gaps)
	    + self.time_to_cutoff_embedding(recency)
        )

        x = self.dropout(self.layer_norm(x))

        x = self.transformer(
            x, src_key_padding_mask=padding_mask
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        pooled = (
            (x * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        return self.classifier(pooled).squeeze(-1)


# ============================================================
# 5. LOAD CHECKPOINT
# ============================================================

model = TimeAwareBEHRT().to(DEVICE)

try:
    state = torch.load(
        "timeaware_behrt_365d_best.pt",
        map_location=DEVICE,
        weights_only=True
    )
except TypeError:
    state = torch.load(
        "behrt_365d_control_best.pt",
        map_location=DEVICE
    )

model.load_state_dict(state, strict=True)
model.eval()

loader = DataLoader(
    BEHRTDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. PREDICTION / MASKING
# ============================================================

def evaluate(mask_code=None, mask_feature=None):

    ids, labels, probabilities = [], [], []

    with torch.no_grad():

        for batch in loader:

            tensors = {
                k: batch[k].to(DEVICE)
                for k in [
                    "codes", "ages", "segments",
                    "positions", "gaps", "recency"
                ]
            }

            if mask_code is not None:
                codes = tensors["codes"].clone()
                codes[codes == mask_code] = 1
                tensors["codes"] = codes

            if mask_feature is not None:
                values = tensors[mask_feature].clone()
                values[tensors["codes"] != 0] = 0
                tensors[mask_feature] = values

            logits = model(
                tensors["codes"],
                tensors["ages"],
                tensors["segments"],
                tensors["positions"],
                tensors["gaps"],
                tensors["recency"]
            )

            probs = torch.sigmoid(logits).cpu().numpy()

            ids.extend(batch["subject_id"].numpy())
            labels.extend(batch["label"].numpy())
            probabilities.extend(probs)

    return (
        np.asarray(ids),
        np.asarray(labels),
        np.asarray(probabilities)
    )


# ============================================================
# 7. VERIFY ORIGINAL BASELINE
# ============================================================

ids, y, p = evaluate()

baseline_auc = roc_auc_score(y, p)
baseline_ap = average_precision_score(y, p)

print(
    f"Baseline AUROC={baseline_auc:.6f}, "
    f"AUPRC={baseline_ap:.6f}",
    flush=True
)

reference = pd.read_csv(
    "timeaware_behrt_365d_predictions.csv"
)

candidates = [
    "y_prob", "predicted_probability",
    "predicted_prob", "predicted_365d_risk",
    "probability", "pred_prob"
]

prob_col = next(
    (c for c in candidates if c in reference.columns),
    None
)

if prob_col is None:
    raise RuntimeError(
        "Cannot identify probability column: "
        + str(reference.columns.tolist())
    )

comparison = pd.DataFrame({
    "subject_id": ids,
    "reproduced": p
}).merge(
    reference[["subject_id", prob_col]],
    on="subject_id",
    how="left",
    validate="one_to_one"
)

if comparison[prob_col].isna().any():
    raise RuntimeError("Missing reference predictions.")

max_diff = np.max(np.abs(
    comparison["reproduced"] - comparison[prob_col]
))

print(
    f"Original probability difference={max_diff:.8f}",
    flush=True
)

if max_diff > 0.001:
    print(
        "WARNING: Original August predictions do not match "
        "the saved checkpoint. Using checkpoint-reproduced "
        "predictions for a separate exploratory RQ6 analysis.",
        flush=True  
    )

pd.DataFrame({
    "subject_id": ids,
    "label": y,
    "probability": p
}).to_csv(
    OUT / "baseline_predictions.csv",
    index=False
)

pd.DataFrame([{
    "AUROC": baseline_auc,
    "AUPRC": baseline_ap,
    "N_test": len(y)
}]).to_csv(
    OUT / "baseline_metrics.csv",
    index=False
)


# ============================================================
# 8. DIAGNOSIS-CODE IMPORTANCE
# ============================================================

counts = Counter()

for sid in test_ids:
    counts.update(
        c for c in patient_data[int(sid)]["codes"]
        if c >= 2
    )

id_to_code = {
    v: k for k, v in code_to_id.items()
}

rows = []

for index, (code_id, count) in enumerate(
    counts.most_common(TOP_N), start=1
):

    _, _, masked = evaluate(mask_code=code_id)

    new_auc = roc_auc_score(y, masked)
    new_ap = average_precision_score(y, masked)

    rows.append({
        "icd_token": id_to_code[code_id],
        "test_token_occurrences": count,
        "test_patient_count": sum(
            code_id in patient_data[int(sid)]["codes"]
            for sid in test_ids
        ),
        "AUROC_drop": baseline_auc - new_auc,
        "AUPRC_drop": baseline_ap - new_ap
    })

    print(
        f"{index}/{TOP_N} {id_to_code[code_id]}: "
        f"AUROC drop {baseline_auc - new_auc:.6f}",
        flush=True
    )

codes = pd.DataFrame(rows).sort_values(
    "AUROC_drop", ascending=False
)

codes.to_csv(
    OUT / "behrt_code_masking_importance.csv",
    index=False
)


# ============================================================
# 9. AGE AND SEGMENT IMPORTANCE
# ============================================================

feature_rows = []

for feature in ["ages", "segments", "gaps", "recency"]:

    _, _, masked = evaluate(mask_feature=feature)

    feature_rows.append({
        "feature": feature,
        "AUROC_drop": (
            baseline_auc - roc_auc_score(y, masked)
        ),
        "AUPRC_drop": (
            baseline_ap - average_precision_score(y, masked)
        )
    })

features = pd.DataFrame(feature_rows)

features.to_csv(
    OUT / "behrt_embedding_masking_importance.csv",
    index=False
)

print("\nEmbedding importance:")
print(features.to_string(index=False), flush=True)


# ============================================================
# 10. CHARTS
# ============================================================

for metric in ["AUROC", "AUPRC"]:

    col = f"{metric}_drop"

    ordered = codes.sort_values(col)

    plt.figure(figsize=(11, 9))
    plt.barh(ordered["icd_token"], ordered[col])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"{metric} reduction")
    plt.title(f"BEHRT Control: {metric} code importance")
    plt.tight_layout()
    plt.savefig(
        OUT / f"behrt_{metric.lower()}_bar.png",
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.barh(features["feature"], features[col])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"{metric} reduction")
    plt.title(f"BEHRT Control: {metric} embedding importance")
    plt.tight_layout()
    plt.savefig(
        OUT / f"behrt_embedding_{metric.lower()}_bar.png",
        dpi=200
    )
    plt.close()

ordered = codes.sort_values(
    "AUROC_drop", ascending=False
)

plt.figure(figsize=(12, 6))
plt.plot(
    ordered["icd_token"],
    ordered["AUROC_drop"],
    marker="o",
    label="AUROC"
)
plt.plot(
    ordered["icd_token"],
    ordered["AUPRC_drop"],
    marker="s",
    label="AUPRC"
)
plt.xticks(rotation=70, ha="right")
plt.axhline(0, color="black", linewidth=0.8)
plt.legend()
plt.ylabel("Performance reduction")
plt.tight_layout()
plt.savefig(
    OUT / "behrt_code_importance_line.png",
    dpi=200
)
plt.close()

print("\nCOMPLETE. Results:", OUT.resolve(), flush=True)
print(codes.head(15).to_string(index=False), flush=True)
print(
    "Masking importance measures predictive sensitivity, "
    "not causality.",
    flush=True
)

