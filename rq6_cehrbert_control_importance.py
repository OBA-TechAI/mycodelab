
import random
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
# 1. SETTINGS
# ============================================================

SEED = 42
MAX_LEN = 256
MAX_VOCAB = 14000
HIDDEN = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1
MAX_AGE = 120
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT = Path("rq6_cehrbert_control_outputs")
OUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE, flush=True)


# ============================================================
# 2. LOAD LEAKAGE-SAFE DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_365d_age.parquet"
)

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

assert "split" in cohort.columns, (
    "Expected a split column named 'split'."
)

train_ids = cohort.loc[
    cohort["split"].astype(str).str.lower() == "train",
    "subject_id"
].tolist()

test_ids = cohort.loc[
    cohort["split"].astype(str).str.lower() == "test",
    "subject_id"
].tolist()

assert train_ids and test_ids, "Empty train/test split."

print("Train:", len(train_ids))
print("Test:", len(test_ids))


# ============================================================
# 3. TRAIN-ONLY VOCABULARY
# ============================================================

train_codes = seq[
    seq["subject_id"].isin(train_ids)
]["icd_token"].value_counts()

top_codes = train_codes.index[
    :MAX_VOCAB - 4
].tolist()

code_to_id = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<VS>": 2,
    "<VE>": 3
}

for i, code in enumerate(top_codes, start=4):
    code_to_id[code] = i

VOCAB_SIZE = len(code_to_id)

id_to_code = {
    value: key
    for key, value in code_to_id.items()
}

print("Vocabulary size:", VOCAB_SIZE)


# ============================================================
# 4. LABELS AND PATIENT SEQUENCES
# ============================================================

label_map = dict(
    zip(
        cohort["subject_id"],
        cohort["dn_within_365d"]
    )
)

seq = seq.sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

patient_data = {}

test_seq = seq[
    seq["subject_id"].isin(test_ids)
]

for sid, patient_df in test_seq.groupby("subject_id"):

    tokens = []
    ages = []
    visit_types = []

    for visit_number, visit_df in patient_df.groupby(
        "visit_number",
        sort=True
    ):

        visit_df = visit_df.sort_values("code_rank")

        age = int(visit_df["visit_age"].iloc[0])
        age = max(0, min(MAX_AGE, age))

        visit_type = (int(visit_number) - 1) % 2

        # Visit-start token
        tokens.append(2)
        ages.append(age)
        visit_types.append(visit_type)

        # Diagnosis codes
        for code in visit_df["icd_token"]:

            tokens.append(
                code_to_id.get(code, 1)
            )
            ages.append(age)
            visit_types.append(visit_type)

        # Visit-end token
        tokens.append(3)
        ages.append(age)
        visit_types.append(visit_type)

    if len(tokens) > MAX_LEN:

        tokens = tokens[-MAX_LEN:]
        ages = ages[-MAX_LEN:]
        visit_types = visit_types[-MAX_LEN:]

    patient_data[int(sid)] = {
        "codes": tokens,
        "ages": ages,
        "visit_types": visit_types
    }

assert set(test_ids) == set(patient_data), (
    "Some test patients lack sequences."
)


# ============================================================
# 5. DATASET
# ============================================================

class CEHRBERTDataset(Dataset):

    def __init__(self, patient_ids):
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        sid = int(self.patient_ids[idx])
        p = patient_data[sid]

        codes = list(p["codes"])
        ages = list(p["ages"])
        visit_types = list(p["visit_types"])

        length = len(codes)
        pad = MAX_LEN - length

        positions = list(range(length))

        codes += [0] * pad
        ages += [0] * pad
        visit_types += [0] * pad
        positions += [0] * pad

        return {
            "subject_id": sid,
            "codes": torch.tensor(codes, dtype=torch.long),
            "ages": torch.tensor(ages, dtype=torch.long),
            "visit_types": torch.tensor(
                visit_types, dtype=torch.long
            ),
            "positions": torch.tensor(
                positions, dtype=torch.long
            ),
            "label": torch.tensor(
                float(label_map[sid]),
                dtype=torch.float32
            )
        }


test_loader = DataLoader(
    CEHRBERTDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. ORIGINAL CEHR-BERT CONTROL ARCHITECTURE
# ============================================================

class CEHRBERTControl(nn.Module):

    def __init__(self):

        super().__init__()

        self.code_embedding = nn.Embedding(
            VOCAB_SIZE, HIDDEN, padding_idx=0
        )

        self.age_embedding = nn.Embedding(
            MAX_AGE + 1, HIDDEN
        )

        self.visit_type_embedding = nn.Embedding(
            2, HIDDEN
        )

        self.position_embedding = nn.Embedding(
            MAX_LEN, HIDDEN
        )

        self.layer_norm = nn.LayerNorm(HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS
        )

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1)
        )

    def forward(
        self,
        codes,
        ages,
        visit_types,
        positions
    ):

        padding_mask = codes.eq(0)

        x = (
            self.code_embedding(codes)
            + self.age_embedding(ages)
            + self.visit_type_embedding(visit_types)
            + self.position_embedding(positions)
        )

        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        valid = (
            (~padding_mask).unsqueeze(-1).float()
        )

        pooled = (
            (x * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        return self.classifier(pooled).squeeze(-1)


# ============================================================
# 7. LOAD ORIGINAL CHECKPOINT
# ============================================================

model = CEHRBERTControl().to(DEVICE)

state = torch.load(
    "cehrbert_365d_control_best.pt",
    map_location=DEVICE
)

model.load_state_dict(state, strict=True)
model.eval()

print("Checkpoint loaded.", flush=True)


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(mask_code=None):

    ids = []
    labels = []
    probabilities = []

    for batch in test_loader:

        codes = batch["codes"].to(DEVICE)

        if mask_code is not None:
            codes = codes.clone()
            codes[codes == mask_code] = 1

        logits = model(
            codes,
            batch["ages"].to(DEVICE),
            batch["visit_types"].to(DEVICE),
            batch["positions"].to(DEVICE)
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
# 9. RECONSTRUCT BASELINE
# ============================================================

ids, y, p = evaluate()

baseline_auc = roc_auc_score(y, p)
baseline_ap = average_precision_score(y, p)

print(
    "Reconstructed baseline AUROC:",
    baseline_auc,
    flush=True
)

print(
    "Reconstructed baseline AUPRC:",
    baseline_ap,
    flush=True
)

pd.DataFrame({
    "subject_id": ids,
    "y_true": y,
    "y_prob": p
}).to_csv(
    OUT / "baseline_predictions.csv",
    index=False
)

pd.DataFrame([{
    "model": "CEHR-BERT Control",
    "n_test": len(y),
    "n_events": int(y.sum()),
    "auroc": baseline_auc,
    "auprc": baseline_ap
}]).to_csv(
    OUT / "baseline_metrics.csv",
    index=False
)


# ============================================================
# 10. VERIFY ORIGINAL PREDICTIONS
# ============================================================

original = pd.read_csv(
    "cehrbert_365d_control_predictions.csv"
)

assert {
    "subject_id",
    "y_true",
    "y_prob"
}.issubset(original.columns), (
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

assert (
    comparison["_merge"] == "both"
).all(), "Patient sets differ."

assert np.array_equal(
    comparison["y_true_reconstructed"].to_numpy(),
    comparison["y_true"].to_numpy()
), "Labels differ."

max_diff = np.max(
    np.abs(
        comparison["y_prob_reconstructed"]
        - comparison["y_prob"]
    )
)

print(
    "Max absolute prediction difference:",
    max_diff,
    flush=True
)

if max_diff > 1e-4:
    raise RuntimeError(
        "Baseline mismatch. Stop before importance analysis."
    )

print("BASELINE VERIFIED", flush=True)


# ============================================================
# 11. TOP 30 DIAGNOSIS CODES
# ============================================================

counts = Counter()
patient_counts = Counter()

for sid in test_ids:

    observed = [
        code
        for code in patient_data[int(sid)]["codes"]
        if code >= 4
    ]

    counts.update(observed)
    patient_counts.update(set(observed))

top_30 = counts.most_common(30)

print(
    "Diagnosis codes selected:",
    len(top_30),
    flush=True
)


# ============================================================
# 12. DIAGNOSIS-CODE MASKING IMPORTANCE
# ============================================================

rows = []

for rank, (code_id, n_tokens) in enumerate(
    top_30,
    start=1
):

    _, _, masked_prob = evaluate(
        mask_code=code_id
    )

    masked_auc = roc_auc_score(
        y, masked_prob
    )

    masked_ap = average_precision_score(
        y, masked_prob
    )

    row = {
        "rank": rank,
        "icd_token": id_to_code[code_id],
        "token_count": n_tokens,
        "patient_count": patient_counts[code_id],
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
        row["icd_token"],
        "AUROC drop:",
        round(row["auroc_drop"], 6),
        "AUPRC drop:",
        round(row["auprc_drop"], 6),
        flush=True
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame(rows).sort_values(
    "auroc_drop",
    ascending=False
)

results.to_csv(
    OUT / "diagnosis_code_masking_importance.csv",
    index=False
)


# ============================================================
# 14. GENERATE FIGURES
# ============================================================

for metric, title, filename in [
    (
        "auroc_drop",
        "AUROC reduction",
        "top15_auroc_importance.png"
    ),
    (
        "auprc_drop",
        "AUPRC reduction",
        "top15_auprc_importance.png"
    )
]:

    plot_data = results.nlargest(
        15, metric
    ).sort_values(metric)

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_data["icd_token"],
        plot_data[metric]
    )

    plt.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.xlabel(title + " after masking")

    plt.title(
        "CEHR-BERT Control: Diagnosis-Code Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180
    )

    plt.close()


print(
    "RQ6 CEHR-BERT CONTROL COMPLETED",
    flush=True
)

print(
    "Saved outputs:",
    OUT.resolve(),
    flush=True
)

