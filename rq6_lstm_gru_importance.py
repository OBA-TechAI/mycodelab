
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

# CPU inference avoids the CUDA driver warning seen previously.
DEVICE = torch.device("cpu")

OUTPUT_DIR = Path("rq6_lstm_gru_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE, flush=True)


# ============================================================
# 2. LOAD ORIGINAL LEAKAGE-SAFE SEQUENCES
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_365d.parquet"
)

seq = seq.sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

assert (
    pd.to_datetime(seq["admittime"])
    < pd.to_datetime(seq["prediction_cutoff"])
).all(), "Pre-cutoff leakage check failed."

patient_data = {}

for sid, group in seq.groupby("subject_id"):
    patient_data[int(sid)] = {
        "tokens": group["icd_token"].astype(str).tolist(),
        "label": int(group["dn_within_365d"].iloc[0]),
        "split": str(group["split"].iloc[0])
    }

train_ids = [
    sid for sid, p in patient_data.items()
    if p["split"] == "train"
]

test_ids = [
    sid for sid, p in patient_data.items()
    if p["split"] == "test"
]

print("Train patients:", len(train_ids), flush=True)
print("Test patients:", len(test_ids), flush=True)


# ============================================================
# 3. REBUILD TRAINING VOCABULARY EXACTLY
# ============================================================

train_tokens = set()

for sid in train_ids:
    train_tokens.update(patient_data[sid]["tokens"])

token_to_id = {
    token: i + 2
    for i, token in enumerate(sorted(train_tokens))
}

VOCAB_SIZE = len(token_to_id) + 2

print("Vocabulary size:", VOCAB_SIZE, flush=True)


# ============================================================
# 4. ORIGINAL DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        sid = self.ids[index]
        patient = patient_data[sid]

        codes = [
            token_to_id.get(token, UNK_ID)
            for token in patient["tokens"]
        ]

        if len(codes) > MAX_LEN:
            codes = codes[-MAX_LEN:]

        length = len(codes)

        codes = codes + [PAD_ID] * (MAX_LEN - length)

        return {
            "subject_id": sid,
            "codes": torch.tensor(codes, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(
                patient["label"], dtype=torch.float32
            )
        }


test_loader = DataLoader(
    SequenceDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. ORIGINAL LSTM / GRU ARCHITECTURE
# ============================================================

class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, rnn_type):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=EMBED_DIM,
                hidden_size=HIDDEN_DIM,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=EMBED_DIM,
                hidden_size=HIDDEN_DIM,
                batch_first=True
            )
        else:
            raise ValueError("Invalid RNN type")

        self.dropout = nn.Dropout(0.20)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)
        self.rnn_type = rnn_type

    def forward(self, codes, lengths):

        x = self.embedding(codes)

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
# 6. PREPARE TEST ARRAYS
# ============================================================

all_codes = []
all_lengths = []
all_labels = []
all_subjects = []

for batch in test_loader:
    all_codes.append(batch["codes"])
    all_lengths.append(batch["length"])
    all_labels.append(batch["label"].numpy())
    all_subjects.extend(batch["subject_id"].tolist())

codes_tensor = torch.cat(all_codes)
lengths_tensor = torch.cat(all_lengths)

y_test = np.concatenate(all_labels).astype(int)
subject_ids = np.asarray(all_subjects)

assert len(subject_ids) == 1337
assert int(y_test.sum()) == 103

# Count diagnosis tokens in the actual truncated test sequences.
frequency = Counter()

for codes, length in zip(
    codes_tensor.tolist(),
    lengths_tensor.tolist()
):
    frequency.update(codes[:length])

id_to_token = {
    value: key for key, value in token_to_id.items()
}

selected_tokens = [
    (token_id, count)
    for token_id, count in frequency.most_common()
    if token_id not in (PAD_ID, UNK_ID)
][:TOP_K]

print(
    "Diagnosis tokens selected:",
    len(selected_tokens),
    flush=True
)


# ============================================================
# 7. BATCH PREDICTION
# ============================================================

@torch.no_grad()
def predict(model, codes):

    model.eval()

    outputs = []

    for start in range(0, len(codes), BATCH_SIZE):

        end = start + BATCH_SIZE

        batch_codes = codes[start:end].to(DEVICE)
        batch_lengths = lengths_tensor[start:end]

        logits = model(
            batch_codes,
            batch_lengths
        )

        probabilities = torch.sigmoid(logits)

        outputs.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(outputs)


# ============================================================
# 8. LOAD MODELS, VERIFY AND MEASURE IMPORTANCE
# ============================================================

all_importance = []
all_baselines = []

for rnn_type in ("LSTM", "GRU"):

    name = rnn_type.lower()

    print(
        "\n" + "=" * 60,
        "\nMODEL:", rnn_type,
        flush=True
    )

    model = RNNClassifier(
        VOCAB_SIZE,
        rnn_type
    ).to(DEVICE)

    checkpoint = torch.load(
        f"{name}_365d_best.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    baseline_prob = predict(
        model,
        codes_tensor
    )

    baseline_auc = roc_auc_score(
        y_test,
        baseline_prob
    )

    baseline_ap = average_precision_score(
        y_test,
        baseline_prob
    )

    print("AUROC:", baseline_auc, flush=True)
    print("AUPRC:", baseline_ap, flush=True)

    reconstructed = pd.DataFrame({
        "subject_id": subject_ids,
        "label": y_test,
        "probability": baseline_prob
    })

    reconstructed.to_csv(
        OUTPUT_DIR / f"rq6_{name}_reconstructed_predictions.csv",
        index=False
    )

    original = pd.read_csv(
        f"{name}_365d_predictions.csv"
    )

    comparison = original.merge(
        reconstructed,
        on="subject_id",
        suffixes=("_original", "_reconstructed"),
        validate="one_to_one"
    )

    assert len(comparison) == len(y_test)
    assert (
        comparison["label_original"]
        == comparison["label_reconstructed"]
    ).all()

    differences = np.abs(
        comparison["probability_original"].to_numpy()
        - comparison["probability_reconstructed"].to_numpy()
    )

    max_diff = float(differences.max())
    mean_diff = float(differences.mean())

    verified = max_diff <= 1e-5

    print(
        "Maximum prediction difference:",
        max_diff,
        flush=True
    )
    print(
        "Mean prediction difference:",
        mean_diff,
        flush=True
    )
    print(
        "Baseline verified:",
        verified,
        flush=True
    )

    all_baselines.append({
        "model": rnn_type,
        "n_test": len(y_test),
        "dn_events": int(y_test.sum()),
        "auroc": baseline_auc,
        "auprc": baseline_ap,
        "max_prediction_difference": max_diff,
        "mean_prediction_difference": mean_diff,
        "verified": verified
    })

    pd.DataFrame(all_baselines).to_csv(
        OUTPUT_DIR / "rq6_lstm_gru_baseline_verification.csv",
        index=False
    )

    if not verified:
        print(
            "Verification failed. Skipping importance for",
            rnn_type,
            flush=True
        )
        continue

    for rank, (token_id, count) in enumerate(
        selected_tokens,
        start=1
    ):

        perturbed_codes = codes_tensor.clone()

        # Replace the selected diagnosis token with UNK.
        # Do not change padding or sequence length.
        perturbed_codes[
            perturbed_codes == token_id
        ] = UNK_ID

        perturbed_prob = predict(
            model,
            perturbed_codes
        )

        perturbed_auc = roc_auc_score(
            y_test,
            perturbed_prob
        )

        perturbed_ap = average_precision_score(
            y_test,
            perturbed_prob
        )

        auc_drop = baseline_auc - perturbed_auc
        ap_drop = baseline_ap - perturbed_ap

        token_name = id_to_token[token_id]

        result = {
            "model": rnn_type,
            "rank_by_frequency": rank,
            "diagnosis_token": token_name,
            "token_id": token_id,
            "token_occurrences": count,
            "replacement": "UNK_ID",
            "baseline_auroc": baseline_auc,
            "perturbed_auroc": perturbed_auc,
            "auroc_drop": auc_drop,
            "baseline_auprc": baseline_ap,
            "perturbed_auprc": perturbed_ap,
            "auprc_drop": ap_drop
        }

        all_importance.append(result)

        print(
            rank,
            token_name,
            "| AUROC drop:",
            round(auc_drop, 6),
            "| AUPRC drop:",
            round(ap_drop, 6),
            flush=True
        )

    pd.DataFrame(all_importance).to_csv(
        OUTPUT_DIR / "rq6_lstm_gru_diagnosis_importance.csv",
        index=False
    )


# ============================================================
# 9. FINISH
# ============================================================

print()
print("RQ6 LSTM/GRU ANALYSIS FINISHED", flush=True)
print("Results saved in:", OUTPUT_DIR.resolve(), flush=True)

