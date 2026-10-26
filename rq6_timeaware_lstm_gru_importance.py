
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
MAX_LEN = 256
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 128
TOP_K = 30

PAD_ID = 0
UNK_ID = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False

DEVICE = torch.device("cpu")

OUT = Path("rq6_timeaware_lstm_gru_outputs")
OUT.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD ORIGINAL DATA
# ============================================================

seq = pd.read_parquet("dx_sequences_timeaware_365d.parquet")
seq = seq.sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

assert (
    pd.to_datetime(seq["admittime"])
    < pd.to_datetime(seq["prediction_cutoff"])
).all()

patients = {}

for sid, g in seq.groupby("subject_id"):
    patients[int(sid)] = {
        "tokens": g["icd_token"].astype(str).tolist(),
        "gaps": g["time_gap_days"].astype(float).tolist(),
        "recency": g["time_to_cutoff_days"].astype(float).tolist(),
        "label": int(g["dn_within_365d"].iloc[0]),
        "split": str(g["split"].iloc[0])
    }

train_ids = [
    sid for sid, p in patients.items()
    if p["split"] == "train"
]

test_ids = [
    sid for sid, p in patients.items()
    if p["split"] == "test"
]

# ============================================================
# 3. EXACT TRAIN-ONLY TEMPORAL SCALING
# ============================================================

train_gaps = np.concatenate([
    np.asarray(patients[sid]["gaps"], dtype=float)
    for sid in train_ids
])

train_recency = np.concatenate([
    np.asarray(patients[sid]["recency"], dtype=float)
    for sid in train_ids
])

gap_log = np.log1p(np.clip(train_gaps, 0, None))
recency_log = np.log1p(np.clip(train_recency, 0, None))

GAP_MEAN = float(gap_log.mean())
GAP_STD = float(gap_log.std()) or 1.0

RECENCY_MEAN = float(recency_log.mean())
RECENCY_STD = float(recency_log.std()) or 1.0

# ============================================================
# 4. ORIGINAL TRAIN-ONLY VOCABULARY
# ============================================================

train_tokens = set()

for sid in train_ids:
    train_tokens.update(patients[sid]["tokens"])

token_to_id = {
    token: i + 2
    for i, token in enumerate(sorted(train_tokens))
}

id_to_token = {
    value: key for key, value in token_to_id.items()
}

VOCAB_SIZE = len(token_to_id) + 2

# ============================================================
# 5. BUILD TEST ARRAYS
# ============================================================

codes_list = []
gaps_list = []
recency_list = []
lengths_list = []
labels = []

for sid in test_ids:
    p = patients[sid]

    codes = [
        token_to_id.get(t, UNK_ID)
        for t in p["tokens"]
    ]

    gaps = np.asarray(p["gaps"], dtype=float)
    recency = np.asarray(p["recency"], dtype=float)

    if len(codes) > MAX_LEN:
        codes = codes[-MAX_LEN:]
        gaps = gaps[-MAX_LEN:]
        recency = recency[-MAX_LEN:]

    length = len(codes)

    gaps = (
        np.log1p(np.clip(gaps, 0, None)) - GAP_MEAN
    ) / GAP_STD

    recency = (
        np.log1p(np.clip(recency, 0, None)) - RECENCY_MEAN
    ) / RECENCY_STD

    pad = MAX_LEN - length

    codes_list.append(codes + [PAD_ID] * pad)
    gaps_list.append(np.pad(gaps, (0, pad)))
    recency_list.append(np.pad(recency, (0, pad)))
    lengths_list.append(length)
    labels.append(p["label"])

codes = torch.tensor(codes_list, dtype=torch.long)
gaps = torch.tensor(
    np.asarray(gaps_list), dtype=torch.float32
)
recency = torch.tensor(
    np.asarray(recency_list), dtype=torch.float32
)
lengths = torch.tensor(lengths_list, dtype=torch.long)

y = np.asarray(labels, dtype=int)

assert len(y) == 1337
assert int(y.sum()) == 103
assert (lengths > 0).all()

print("Test patients:", len(y), flush=True)
print("DN events:", int(y.sum()), flush=True)
print("Vocabulary:", VOCAB_SIZE, flush=True)

# ============================================================
# 6. ORIGINAL TIME-AWARE ARCHITECTURE
# ============================================================

class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, rnn_type):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        input_dim = EMBED_DIM + 2

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=HIDDEN_DIM,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=HIDDEN_DIM,
                batch_first=True
            )
        else:
            raise ValueError("Invalid RNN type")

        self.dropout = nn.Dropout(0.20)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)
        self.rnn_type = rnn_type

    def forward(self, codes, gaps, recency, lengths):

        embeddings = self.embedding(codes)

        temporal = torch.stack(
            [gaps, recency],
            dim=-1
        )

        x = torch.cat([embeddings, temporal], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.rnn_type == "LSTM":
            _, (hidden, _) = self.rnn(packed)
        else:
            _, hidden = self.rnn(packed)

        h = self.dropout(hidden[-1])

        return self.classifier(h).squeeze(-1)


# ============================================================
# 7. PREDICTION
# ============================================================

@torch.no_grad()
def predict(model, input_codes, input_gaps, input_recency):

    model.eval()
    outputs = []

    for start in range(0, len(y), BATCH_SIZE):
        end = start + BATCH_SIZE

        logits = model(
            input_codes[start:end],
            input_gaps[start:end],
            input_recency[start:end],
            lengths[start:end]
        )

        outputs.append(
            torch.sigmoid(logits).numpy()
        )

    return np.concatenate(outputs)


def metrics(prob):
    return (
        roc_auc_score(y, prob),
        average_precision_score(y, prob)
    )


# ============================================================
# 8. SELECT TOP FREQUENT DIAGNOSES
# ============================================================

frequency = Counter()

for row, length in zip(
    codes.tolist(),
    lengths.tolist()
):
    frequency.update(row[:length])

top_tokens = [
    (token_id, count)
    for token_id, count in frequency.most_common()
    if token_id not in (PAD_ID, UNK_ID)
][:TOP_K]


# ============================================================
# 9. MODEL VERIFICATION AND SENSITIVITY
# ============================================================

verification = []
importance = []

for rnn_type in ["LSTM", "GRU"]:

    name = rnn_type.lower()

    print("\nMODEL:", rnn_type, flush=True)

    model = RNNClassifier(
        VOCAB_SIZE, rnn_type
    ).to(DEVICE)

    state = torch.load(
        f"{name}_timeaware_365d_best.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    baseline_prob = predict(
        model, codes, gaps, recency
    )

    baseline_auc, baseline_ap = metrics(baseline_prob)

    original = pd.read_csv(
        f"{name}_timeaware_365d_predictions.csv"
    )

    reconstructed = pd.DataFrame({
        "subject_id": test_ids,
        "label": y,
        "probability": baseline_prob
    })

    comparison = original.merge(
        reconstructed,
        on="subject_id",
        suffixes=("_original", "_reconstructed"),
        validate="one_to_one"
    )

    assert len(comparison) == len(y)
    assert (
        comparison["label_original"]
        == comparison["label_reconstructed"]
    ).all()

    differences = np.abs(
        comparison["probability_original"].to_numpy()
        - comparison["probability_reconstructed"].to_numpy()
    )

    max_diff = float(differences.max())
    verified = max_diff <= 1e-5

    print("AUROC:", baseline_auc, flush=True)
    print("AUPRC:", baseline_ap, flush=True)
    print("Maximum prediction difference:", max_diff, flush=True)
    print("Verified:", verified, flush=True)

    verification.append({
        "model": rnn_type,
        "auroc": baseline_auc,
        "auprc": baseline_ap,
        "max_prediction_difference": max_diff,
        "verified": verified
    })

    pd.DataFrame(verification).to_csv(
        OUT / "baseline_verification.csv",
        index=False
    )

    if not verified:
        print("Skipping importance: verification failed.")
        continue

    # --------------------------------------------------------
    # A. DIAGNOSIS TOKEN REPLACEMENT
    # --------------------------------------------------------

    for token_id, count in top_tokens:

        modified = codes.clone()
        modified[modified == token_id] = UNK_ID

        prob = predict(
            model, modified, gaps, recency
        )

        auc, ap = metrics(prob)

        importance.append({
            "model": rnn_type,
            "component": "diagnosis",
            "feature": id_to_token[token_id],
            "frequency": count,
            "auroc_drop": baseline_auc - auc,
            "auprc_drop": baseline_ap - ap
        })

    # --------------------------------------------------------
    # B. TIME-GAP AND RECENCY ABLATIONS
    # --------------------------------------------------------

    # Zero in standardised space = training-set mean
    # in log-transformed space, not zero elapsed days.
    valid = (
        torch.arange(MAX_LEN)[None, :]
        < lengths[:, None]
    )

    for component in ["gap", "recency", "both_temporal"]:

        modified_gaps = gaps.clone()
        modified_recency = recency.clone()

        if component in ["gap", "both_temporal"]:
            modified_gaps[valid] = 0.0

        if component in ["recency", "both_temporal"]:
            modified_recency[valid] = 0.0

        prob = predict(
            model,
            codes,
            modified_gaps,
            modified_recency
        )

        auc, ap = metrics(prob)

        importance.append({
            "model": rnn_type,
            "component": component,
            "feature": "all_valid_events",
            "frequency": int(valid.sum()),
            "auroc_drop": baseline_auc - auc,
            "auprc_drop": baseline_ap - ap
        })

        print(
            component,
            "| AUROC drop:",
            round(baseline_auc - auc, 6),
            "| AUPRC drop:",
            round(baseline_ap - ap, 6),
            flush=True
        )

    pd.DataFrame(importance).to_csv(
        OUT / "diagnosis_temporal_importance.csv",
        index=False
    )

    print(
        "Diagnosis tokens analysed:",
        len(top_tokens),
        flush=True
    )


print("\nRQ6 TIME-AWARE LSTM/GRU COMPLETED", flush=True)
print("Outputs:", OUT.resolve(), flush=True)

