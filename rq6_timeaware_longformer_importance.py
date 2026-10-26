
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
MODEL_PATH = "./Clinical-Longformer"
CHECKPOINT = "timeaware_clinical_longformer_365d_best.pt"
ORIGINAL_PREDICTIONS = (
    "timeaware_clinical_longformer_365d_predictions.csv"
)

MAX_LENGTH = 1024
BATCH_SIZE = 2
TOP_N = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT = Path("rq6_timeaware_longformer_outputs")
OUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE, flush=True)


# ============================================================
# 2. LOAD ORIGINAL DATA AND SPLITS
# ============================================================

seq = pd.read_parquet("dx_sequences_biobert_365d.parquet")

cohort = pd.read_parquet(
    "cohort_labels_timeaware_365d.parquet"
)[["subject_id", "dn_within_365d"]]

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

test_ids = cohort.loc[
    cohort["split"] == "test", "subject_id"
].tolist()

label_map = dict(zip(
    cohort["subject_id"],
    cohort["dn_within_365d"]
))

print("Test patients:", len(test_ids), flush=True)


# ============================================================
# 3. RECONSTRUCT PATIENT VISITS
# ============================================================

seq = seq.sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

patient_visits = {}

for sid, patient_df in seq[
    seq["subject_id"].isin(test_ids)
].groupby("subject_id"):

    visits = []

    for visit_number, visit_df in patient_df.groupby(
        "visit_number", sort=True
    ):

        visit_df = visit_df.sort_values("code_rank")

        diagnoses = (
            visit_df["diagnosis_text"]
            .astype(str)
            .tolist()
        )

        recency_days = int(round(float(
            visit_df["time_to_cutoff_days"].iloc[0]
        )))

        gap_days = int(round(float(
            visit_df["time_gap_days"].iloc[0]
        )))

        visits.append((
            int(visit_number),
            diagnoses,
            recency_days,
            gap_days
        ))

    patient_visits[int(sid)] = visits

assert set(test_ids) == set(patient_visits), (
    "Missing patient histories."
)


# ============================================================
# 4. BUILD TEXT WITH OPTIONAL MASKING
# ============================================================

def make_text(
    sid,
    mask_diagnosis=None,
    mask_recency=False,
    mask_gaps=False
):

    visit_texts = []

    for visit_number, diagnoses, recency, gap in patient_visits[int(sid)]:

        current = [
            "[MASK]" if diagnosis == mask_diagnosis
            else diagnosis
            for diagnosis in diagnoses
        ]

        if mask_recency:
            recency = 0

        if mask_gaps:
            gap = 0

        if visit_number == 1:
            temporal_text = (
                str(recency)
                + " days before prediction cutoff"
            )
        else:
            temporal_text = (
                str(recency)
                + " days before prediction cutoff; "
                + str(gap)
                + " days since previous visit"
            )

        visit_text = (
            "Visit "
            + str(visit_number)
            + " ["
            + temporal_text
            + "] diagnoses: "
            + "; ".join(current)
            + "."
        )

        visit_texts.append(visit_text)

    return (
        "Patient prior diagnosis history. "
        + " ".join(visit_texts)
    )


# ============================================================
# 5. LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    local_files_only=True
)

state = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state, strict=True)
model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded.", flush=True)


# ============================================================
# 6. DATASET
# ============================================================

class ClinicalBERTDataset(Dataset):

    def __init__(
        self,
        patient_ids,
        mask_diagnosis=None,
        mask_recency=False,
        mask_gaps=False
    ):
        self.patient_ids = patient_ids
        self.mask_diagnosis = mask_diagnosis
        self.mask_recency = mask_recency
        self.mask_gaps = mask_gaps

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):

        sid = int(self.patient_ids[index])

        text = make_text(
            sid,
            mask_diagnosis=self.mask_diagnosis,
            mask_recency=self.mask_recency,
            mask_gaps=self.mask_gaps
        )

        encoded = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "subject_id": sid,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": float(label_map[sid])
        }


# ============================================================
# 7. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    mask_diagnosis=None,
    mask_recency=False,
    mask_gaps=False
):

    dataset = ClinicalBERTDataset(
        test_ids,
        mask_diagnosis=mask_diagnosis,
        mask_recency=mask_recency,
        mask_gaps=mask_gaps
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    ids, labels, probabilities = [], [], []

    for batch in loader:

        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE)
        )

        probs = torch.sigmoid(
            outputs.logits.squeeze(-1)
        ).cpu().numpy()

        ids.extend(batch["subject_id"].numpy())
        labels.extend(batch["label"].numpy())
        probabilities.extend(
            np.asarray(probs).reshape(-1)
        )

    return (
        np.asarray(ids),
        np.asarray(labels),
        np.asarray(probabilities)
    )


# ============================================================
# 8. RECONSTRUCT BASELINE
# ============================================================

ids, y, p = evaluate()

baseline_auc = roc_auc_score(y, p)
baseline_ap = average_precision_score(y, p)

print("Reconstructed baseline AUROC:", baseline_auc, flush=True)
print("Reconstructed baseline AUPRC:", baseline_ap, flush=True)

pd.DataFrame({
    "subject_id": ids,
    "y_true": y,
    "y_prob": p
}).to_csv(
    OUT / "baseline_predictions.csv",
    index=False
)

pd.DataFrame([{
    "model": "Time-Aware Clinical Longformer Diagnosis-Text",
    "n_test": len(y),
    "n_events": int(y.sum()),
    "auroc": baseline_auc,
    "auprc": baseline_ap
}]).to_csv(
    OUT / "baseline_metrics.csv",
    index=False
)


# ============================================================
# 9. VERIFY ORIGINAL PREDICTIONS
# ============================================================

original = pd.read_csv(ORIGINAL_PREDICTIONS)

required = {"subject_id", "y_true", "y_prob"}

assert required.issubset(original.columns), (
    "Original prediction columns differ."
)

comparison = pd.DataFrame({
    "subject_id": ids,
    "y_true_reconstructed": y,
    "y_prob_reconstructed": p
}).merge(
    original[["subject_id", "y_true", "y_prob"]],
    on="subject_id",
    how="outer",
    validate="one_to_one",
    indicator=True
)

assert (comparison["_merge"] == "both").all(), (
    "Patient sets differ."
)

assert np.array_equal(
    comparison["y_true_reconstructed"].to_numpy(),
    comparison["y_true"].to_numpy()
), "Labels differ."

max_diff = np.max(np.abs(
    comparison["y_prob_reconstructed"]
    - comparison["y_prob"]
))

print(
    "Max absolute prediction difference:",
    max_diff,
    flush=True
)

if max_diff > 1e-4:
    raise RuntimeError(
        "Baseline mismatch: importance analysis stopped."
    )

print("BASELINE VERIFIED", flush=True)


# ============================================================
# 10. SELECT 30 FREQUENT DIAGNOSES
# ============================================================

counts = Counter()
patient_counts = Counter()

for sid in test_ids:

    descriptions = [
        diagnosis
        for _, diagnoses, _, _ in patient_visits[int(sid)]
        for diagnosis in diagnoses
    ]

    counts.update(descriptions)
    patient_counts.update(set(descriptions))

top_diagnoses = counts.most_common(TOP_N)

print(
    "Diagnosis descriptions selected:",
    len(top_diagnoses),
    flush=True
)


# ============================================================
# 11. DIAGNOSIS-TEXT MASKING
# ============================================================

rows = []

for rank, (diagnosis, count) in enumerate(
    top_diagnoses, start=1
):

    _, _, masked_prob = evaluate(
        mask_diagnosis=diagnosis
    )

    masked_auc = roc_auc_score(y, masked_prob)
    masked_ap = average_precision_score(y, masked_prob)

    row = {
        "rank_by_frequency": rank,
        "diagnosis_description": diagnosis,
        "occurrence_count": count,
        "patient_count": patient_counts[diagnosis],
        "baseline_auroc": baseline_auc,
        "masked_auroc": masked_auc,
        "auroc_drop": baseline_auc - masked_auc,
        "baseline_auprc": baseline_ap,
        "masked_auprc": masked_ap,
        "auprc_drop": baseline_ap - masked_ap
    }

    rows.append(row)

    print(
        rank,
        diagnosis[:75],
        "AUROC drop:",
        round(row["auroc_drop"], 6),
        "AUPRC drop:",
        round(row["auprc_drop"], 6),
        flush=True
    )

pd.DataFrame(rows).sort_values(
    "auroc_drop", ascending=False
).to_csv(
    OUT / "diagnosis_text_masking_importance.csv",
    index=False
)


# ============================================================
# 12. TEMPORAL-TEXT MASKING
# ============================================================

temporal_rows = []

for feature, kwargs in [
    ("recency_days", {"mask_recency": True}),
    ("visit_gap_days", {"mask_gaps": True})
]:

    _, _, masked_prob = evaluate(**kwargs)

    masked_auc = roc_auc_score(y, masked_prob)
    masked_ap = average_precision_score(y, masked_prob)

    row = {
        "temporal_feature": feature,
        "baseline_auroc": baseline_auc,
        "masked_auroc": masked_auc,
        "auroc_drop": baseline_auc - masked_auc,
        "baseline_auprc": baseline_ap,
        "masked_auprc": masked_ap,
        "auprc_drop": baseline_ap - masked_ap
    }

    temporal_rows.append(row)

    print(
        "Temporal feature:",
        feature,
        "AUROC drop:",
        round(row["auroc_drop"], 6),
        "AUPRC drop:",
        round(row["auprc_drop"], 6),
        flush=True
    )

pd.DataFrame(temporal_rows).to_csv(
    OUT / "temporal_text_masking_importance.csv",
    index=False
)


# ============================================================
# 13. FIGURES
# ============================================================

results = pd.DataFrame(rows)

for metric, filename, xlabel in [
    (
        "auroc_drop",
        "top15_auroc_importance.png",
        "AUROC reduction after masking"
    ),
    (
        "auprc_drop",
        "top15_auprc_importance.png",
        "AUPRC reduction after masking"
    )
]:

    plot_data = results.nlargest(
        15, metric
    ).sort_values(metric)

    plt.figure(figsize=(11, 7))

    plt.barh(
        [
            value[:55]
            for value in plot_data["diagnosis_description"]
        ],
        plot_data[metric]
    )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(xlabel)
    plt.title("Time-Aware Clinical Longformer: Diagnosis Importance")

    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


temporal = pd.DataFrame(temporal_rows)

plt.figure(figsize=(7, 4))
plt.bar(
    temporal["temporal_feature"],
    temporal["auroc_drop"]
)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("AUROC reduction after masking")
plt.title("Time-Aware Clinical Longformer: Temporal Importance")
plt.tight_layout()
plt.savefig(
    OUT / "temporal_auroc_importance.png",
    dpi=180
)
plt.close()


print(
    "RQ6 TIME-AWARE LONGFORMER COMPLETED",
    flush=True
)

print("Saved outputs:", OUT.resolve(), flush=True)

