
"""RQ6 Time-Aware Med-BERT: diagnosis and temporal masking importance."""

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

SEED = 42
MAX_LEN = 256
MAX_VOCAB = 14000
HIDDEN = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1
MAX_VISITS = 64
MAX_RANK = 128
BATCH_SIZE = 32
TOP_N = 30

OUTPUT = Path("rq6_timeaware_medbert_outputs")
OUTPUT.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE, flush=True)


# ============================================================
# 2. LOAD DATA AND REBUILD ORIGINAL VOCABULARY
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

print(
    "Vocabulary:", VOCAB_SIZE,
    "Test patients:", len(test_ids),
    flush=True
)

label_map = dict(
    zip(cohort["subject_id"], cohort["dn_within_365d"])
)

seq = seq.sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

patient_data = {}

for sid, g in seq.groupby("subject_id", sort=False):

    if len(g) > MAX_LEN:
        g = g.iloc[-MAX_LEN:]

    patient_data[int(sid)] = {
        "codes": [
            code_to_id.get(c, 1)
            for c in g["icd_token"]
        ],
        "visits": (
            g["visit_number"].fillna(1)
            .astype(int).clip(1, MAX_VISITS).tolist()
        ),
        "ranks": (
            g["code_rank"].fillna(1)
            .astype(int).clip(1, MAX_RANK).tolist()
        ),
        "gaps": (
            g["time_gap_bucket"].fillna(0)
            .astype(int).clip(0, 8).tolist()
        ),
        "recency": (
            g["time_to_cutoff_bucket"].fillna(0)
            .astype(int).clip(0, 8).tolist()
        )
    }


# ============================================================
# 3. TEST DATASET
# ============================================================

class PatientDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        sid = int(self.ids[index])
        p = patient_data[sid]

        length = len(p["codes"])
        pad = MAX_LEN - length

        return {
            "subject_id": sid,
            "codes": torch.tensor(
                p["codes"] + [0] * pad, dtype=torch.long
            ),
            "visits": torch.tensor(
                p["visits"] + [0] * pad, dtype=torch.long
            ),
            "ranks": torch.tensor(
                p["ranks"] + [0] * pad, dtype=torch.long
            ),
            "gaps": torch.tensor(
                p["gaps"] + [0] * pad, dtype=torch.long
            ),
            "recency": torch.tensor(
                p["recency"] + [0] * pad, dtype=torch.long
            ),
            "positions": torch.tensor(
                list(range(length)) + [0] * pad,
                dtype=torch.long
            ),
            "label": int(label_map[sid])
        }


# ============================================================
# 4. ORIGINAL TIME-AWARE MED-BERT ARCHITECTURE
# ============================================================

class MedBERTStyleClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.code_embedding = nn.Embedding(
            VOCAB_SIZE, HIDDEN, padding_idx=0
        )

        self.visit_embedding = nn.Embedding(
            MAX_VISITS + 1, HIDDEN, padding_idx=0
        )

        self.rank_embedding = nn.Embedding(
            MAX_RANK + 1, HIDDEN, padding_idx=0
        )

        self.position_embedding = nn.Embedding(
            MAX_LEN, HIDDEN
        )

        self.time_gap_embedding = nn.Embedding(
            9, HIDDEN
        )

        self.time_to_cutoff_embedding = nn.Embedding(
            9, HIDDEN
        )

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
        self, codes, visits, ranks,
        gaps, recency, positions
    ):

        padding_mask = codes.eq(0)

        x = (
            self.code_embedding(codes)
            + self.visit_embedding(visits)
            + self.rank_embedding(ranks)
            + self.time_gap_embedding(gaps)
            + self.time_to_cutoff_embedding(recency)
            + self.position_embedding(positions)
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
# 5. LOAD SAVED MODEL — NO RETRAINING
# ============================================================

model = MedBERTStyleClassifier().to(DEVICE)

try:
    state = torch.load(
        "timeaware_medbert_365d_best.pt",
        map_location=DEVICE,
        weights_only=True
    )
except TypeError:
    state = torch.load(
        "timeaware_medbert_365d_best.pt",
        map_location=DEVICE
    )

model.load_state_dict(state, strict=True)
model.eval()

loader = DataLoader(
    PatientDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. PREDICTION AND MASKING FUNCTION
# ============================================================

def evaluate(
    mask_id=None,
    mask_temporal=None,
    mask_bucket=None
):

    ids = []
    ys = []
    probs = []

    with torch.no_grad():

        for batch in loader:

            codes = batch["codes"].to(DEVICE)

            if mask_id is not None:
                codes = codes.clone()
                codes[codes == mask_id] = 1

            gaps = batch["gaps"].to(DEVICE)
            recency = batch["recency"].to(DEVICE)

            if mask_temporal is not None:

                target = (
                    gaps if mask_temporal == "gaps"
                    else recency
                )

                target = target.clone()
                valid = codes.ne(0)

                if mask_bucket is None:
                    target[valid] = 0
                else:
                    selected = (
                        (target == mask_bucket) & valid
                    )
                    target[selected] = 0

                if mask_temporal == "gaps":
                    gaps = target
                else:
                    recency = target

            logits = model(
                codes,
                batch["visits"].to(DEVICE),
                batch["ranks"].to(DEVICE),
                gaps,
                recency,
                batch["positions"].to(DEVICE)
            )

            p = torch.sigmoid(logits).cpu().numpy()

            ids.extend(
                batch["subject_id"].numpy().tolist()
            )
            ys.extend(
                batch["label"].numpy().tolist()
            )
            probs.extend(p.tolist())

    return (
        np.asarray(ids),
        np.asarray(ys),
        np.asarray(probs)
    )


# ============================================================
# 7. BASELINE PERFORMANCE AND VERIFICATION
# ============================================================

ids, y, prob = evaluate()

auc = roc_auc_score(y, prob)
ap = average_precision_score(y, prob)

print(
    f"Baseline AUROC={auc:.6f}, "
    f"AUPRC={ap:.6f}",
    flush=True
)

reference = pd.read_csv(
    "timeaware_medbert_365d_predictions.csv"
)

if "subject_id" not in reference.columns:
    raise RuntimeError(
        "Original predictions lack subject_id"
    )

prob_candidates = [
    "predicted_probability",
    "predicted_prob",
    "predicted_365d_risk",
    "probability",
    "y_prob",
    "pred_prob",
    "predicted_risk",
    "risk_score"
]

prob_col = next(
    (c for c in prob_candidates if c in reference.columns),
    None
)

if prob_col is None:
    prob_col = next(
        (
            c for c in reference.columns
            if any(
                k in c.lower() for k in ["prob", "risk"]
            )
            and pd.api.types.is_numeric_dtype(reference[c])
        ),
        None
    )

if prob_col is None:
    raise RuntimeError(
        f"Cannot identify probability column: "
        f"{list(reference.columns)}"
    )

check = pd.DataFrame({
    "subject_id": ids,
    "reproduced_probability": prob
}).merge(
    reference[["subject_id", prob_col]],
    on="subject_id",
    how="left",
    validate="one_to_one"
)

if check[prob_col].isna().any():
    raise RuntimeError(
        "Original predictions missing test patients"
    )

max_diff = np.max(
    np.abs(
        check["reproduced_probability"]
        - check[prob_col]
    )
)

print(
    f"Original column={prob_col}; "
    f"maximum probability difference={max_diff:.8f}",
    flush=True
)

if max_diff > 0.001:
    raise RuntimeError(
        "Baseline mismatch. Attribution stopped. "
        "Check original preprocessing and vocabulary."
    )

pd.DataFrame({
    "subject_id": ids,
    "dn_within_365d": y,
    "reproduced_probability": prob
}).to_csv(
    OUTPUT / "baseline_predictions.csv",
    index=False
)

pd.DataFrame([{
    "AUROC": auc,
    "AUPRC": ap,
    "N_test": len(y)
}]).to_csv(
    OUTPUT / "baseline_metrics.csv",
    index=False
)


# ============================================================
# 8. DIAGNOSIS CODE IMPORTANCE
# ============================================================

counts = Counter()

for sid in test_ids:
    counts.update(
        c for c in patient_data[int(sid)]["codes"]
        if c >= 2
    )

ranked = counts.most_common(TOP_N)

id_to_code = {
    value: key for key, value in code_to_id.items()
}

rows = []

for idx, (code_id, n_tokens) in enumerate(ranked, 1):

    _, _, masked_prob = evaluate(mask_id=code_id)

    masked_auc = roc_auc_score(y, masked_prob)
    masked_ap = average_precision_score(y, masked_prob)

    rows.append({
        "icd_token": id_to_code[code_id],
        "token_id": code_id,
        "test_token_occurrences": n_tokens,
        "test_patient_count": sum(
            code_id in patient_data[int(sid)]["codes"]
            for sid in test_ids
        ),
        "AUROC_drop": auc - masked_auc,
        "AUPRC_drop": ap - masked_ap,
        "masked_AUROC": masked_auc,
        "masked_AUPRC": masked_ap
    })

    print(
        f"{idx}/{len(ranked)} "
        f"{id_to_code[code_id]}: "
        f"AUROC drop {auc - masked_auc:.6f}",
        flush=True
    )

result = pd.DataFrame(rows).sort_values(
    "AUROC_drop", ascending=False
)

result.to_csv(
    OUTPUT / "timeaware_medbert_code_masking_importance.csv",
    index=False
)


# ============================================================
# 9. DIAGNOSIS CODE CHARTS
# ============================================================

for metric in ["AUROC", "AUPRC"]:

    column = f"{metric}_drop"
    ordered = result.sort_values(column)

    plt.figure(figsize=(11, 9))
    plt.barh(
        ordered["icd_token"].astype(str),
        ordered[column]
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"{metric} reduction after code masking")
    plt.title(
        f"Time-Aware Med-BERT: {metric} code importance"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT / f"timeaware_medbert_{metric.lower()}_bar.png",
        dpi=200
    )
    plt.close()

ordered = result.sort_values(
    "AUROC_drop", ascending=False
)

plt.figure(figsize=(12, 6))

plt.plot(
    ordered["icd_token"].astype(str),
    ordered["AUROC_drop"],
    marker="o",
    label="AUROC"
)

plt.plot(
    ordered["icd_token"].astype(str),
    ordered["AUPRC_drop"],
    marker="s",
    label="AUPRC"
)

plt.xticks(rotation=70, ha="right")
plt.axhline(0, color="black", linewidth=0.8)
plt.legend()
plt.ylabel("Performance reduction")
plt.title("Time-Aware Med-BERT: Code masking")
plt.tight_layout()
plt.savefig(
    OUTPUT / "timeaware_medbert_code_importance_line.png",
    dpi=200
)
plt.close()


# ============================================================
# 10. TEMPORAL IMPORTANCE
# ============================================================

rows_temporal = []

for field, feature in [
    ("gaps", "time_gap_bucket"),
    ("recency", "time_to_cutoff_bucket")
]:

    for bucket in [None, 1, 2, 3, 4, 5, 6, 7, 8]:

        _, _, masked_prob = evaluate(
            mask_temporal=field,
            mask_bucket=bucket
        )

        masked_auc = roc_auc_score(y, masked_prob)
        masked_ap = average_precision_score(
            y, masked_prob
        )

        rows_temporal.append({
            "temporal_feature": feature,
            "mask_scope": (
                "all_valid_events"
                if bucket is None
                else f"bucket_{bucket}_only"
            ),
            "AUROC_drop": auc - masked_auc,
            "AUPRC_drop": ap - masked_ap,
            "masked_AUROC": masked_auc,
            "masked_AUPRC": masked_ap
        })

        print(
            f"Temporal {feature}, "
            f"bucket={bucket}: "
            f"AUROC drop {auc - masked_auc:.6f}",
            flush=True
        )

temporal = pd.DataFrame(rows_temporal)

temporal.to_csv(
    OUTPUT / "timeaware_medbert_temporal_masking_importance.csv",
    index=False
)


# ============================================================
# 11. TEMPORAL CHARTS
# ============================================================

for metric in ["AUROC", "AUPRC"]:

    sub = temporal[
        temporal["mask_scope"] == "all_valid_events"
    ]

    plt.figure(figsize=(8, 4))
    plt.barh(
        sub["temporal_feature"],
        sub[f"{metric}_drop"]
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(
        f"{metric} reduction after temporal masking"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT /
        f"timeaware_medbert_temporal_{metric.lower()}_bar.png",
        dpi=200
    )
    plt.close()

plt.figure(figsize=(10, 5))

for feature, group in temporal[
    temporal["mask_scope"] != "all_valid_events"
].groupby("temporal_feature"):

    plt.plot(
        range(1, 9),
        group["AUROC_drop"].to_numpy(),
        marker="o",
        label=feature
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Original temporal bucket set to zero")
plt.ylabel("AUROC reduction")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "timeaware_medbert_temporal_bucket_line.png",
    dpi=200
)
plt.close()


# ============================================================
# 12. COMPLETION
# ============================================================

print(
    "\nCOMPLETE. Results:",
    OUTPUT.resolve(),
    flush=True
)

print(
    result.head(15).to_string(index=False),
    flush=True
)

print(
    "NOTE: Masking effects describe predictive sensitivity, "
    "not causality. Only the 30 most frequent diagnosis "
    "codes were examined.",
    flush=True
)

