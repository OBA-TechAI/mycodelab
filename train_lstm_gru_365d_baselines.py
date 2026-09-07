import random
import copy
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
    brier_score_loss
)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
MAX_LEN = 256
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Required on current Wolffe PyTorch/CUDA stack for RNN stability
torch.backends.cudnn.enabled = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("cuDNN enabled:", torch.backends.cudnn.enabled)


# ============================================================
# 2. LOAD FINAL LEAKAGE-SAFE SEQUENCES
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_365d.parquet"
)

seq = seq.sort_values(
    [
        "subject_id",
        "visit_number",
        "code_rank"
    ]
)

print("\nRows:", len(seq))
print("Patients:", seq["subject_id"].nunique())
print(
    "All events before cutoff:",
    bool(
        (
            pd.to_datetime(seq["admittime"])
            < pd.to_datetime(seq["prediction_cutoff"])
        ).all()
    )
)


# ============================================================
# 3. BUILD PATIENT SEQUENCES
# ============================================================

patient_data = {}

for sid, g in seq.groupby("subject_id"):

    patient_data[int(sid)] = {
        "tokens": g["icd_token"].astype(str).tolist(),
        "label": int(g["dn_within_365d"].iloc[0]),
        "split": str(g["split"].iloc[0])
    }

train_ids = [
    sid for sid, p in patient_data.items()
    if p["split"] == "train"
]

val_ids = [
    sid for sid, p in patient_data.items()
    if p["split"] == "validation"
]

test_ids = [
    sid for sid, p in patient_data.items()
    if p["split"] == "test"
]

print("\nTrain:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# 4. TRAIN-ONLY VOCABULARY
# ============================================================

train_tokens = set()

for sid in train_ids:
    train_tokens.update(
        patient_data[sid]["tokens"]
    )

PAD_ID = 0
UNK_ID = 1

token_to_id = {
    token: i + 2
    for i, token in enumerate(
        sorted(train_tokens)
    )
}

VOCAB_SIZE = len(token_to_id) + 2

print("\nTraining vocabulary size:", VOCAB_SIZE)


# ============================================================
# 5. DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        sid = self.ids[idx]
        p = patient_data[sid]

        ids = [
            token_to_id.get(
                token,
                UNK_ID
            )
            for token in p["tokens"]
        ]

        # Keep most recent MAX_LEN diagnosis events
        if len(ids) > MAX_LEN:
            ids = ids[-MAX_LEN:]

        length = len(ids)

        ids = ids + (
            [PAD_ID] * (MAX_LEN - length)
        )

        return {
            "subject_id": sid,
            "codes": torch.tensor(
                ids,
                dtype=torch.long
            ),
            "length": torch.tensor(
                length,
                dtype=torch.long
            ),
            "label": torch.tensor(
                p["label"],
                dtype=torch.float32
            )
        }


train_loader = DataLoader(
    SequenceDataset(train_ids),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SequenceDataset(val_ids),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    SequenceDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. LSTM / GRU MODEL
# ============================================================

class RNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        rnn_type
    ):
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
            raise ValueError(
                "rnn_type must be LSTM or GRU"
            )

        self.dropout = nn.Dropout(0.20)

        self.classifier = nn.Linear(
            HIDDEN_DIM,
            1
        )

        self.rnn_type = rnn_type

    def forward(
        self,
        codes,
        lengths
    ):

        x = self.embedding(codes)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.rnn_type == "LSTM":

            _, (hidden, _) = self.rnn(
                packed
            )

        else:

            _, hidden = self.rnn(
                packed
            )

        h = hidden[-1]

        h = self.dropout(h)

        return self.classifier(
            h
        ).squeeze(-1)


# ============================================================
# 7. PREDICTION
# ============================================================

def predict(model, loader):

    model.eval()

    subject_ids = []
    labels = []
    probabilities = []

    with torch.no_grad():

        for batch in loader:

            logits = model(
                batch["codes"].to(DEVICE),
                batch["length"]
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
                probs.cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(subject_ids),
        np.asarray(labels, dtype=int),
        np.asarray(probabilities)
    )


# ============================================================
# 8. THRESHOLD + EVALUATION
# ============================================================

def select_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.05,
        0.951,
        0.005
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def evaluate(
    model_name,
    val_labels,
    val_probs,
    test_labels,
    test_probs
):

    threshold = select_threshold(
        val_labels,
        val_probs
    )

    predictions = (
        test_probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "Model": model_name,
        "N_test": len(test_labels),
        "AUROC": roc_auc_score(
            test_labels,
            test_probs
        ),
        "AUPRC": average_precision_score(
            test_labels,
            test_probs
        ),
        "Accuracy": accuracy_score(
            test_labels,
            predictions
        ),
        "Precision": precision_score(
            test_labels,
            predictions,
            zero_division=0
        ),
        "Recall_Sensitivity": recall_score(
            test_labels,
            predictions,
            zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(
            test_labels,
            predictions,
            zero_division=0
        ),
        "Brier": brier_score_loss(
            test_labels,
            test_probs
        ),
        "Threshold": threshold,
        "Val_AUROC": roc_auc_score(
            val_labels,
            val_probs
        ),
        "Val_AUPRC": average_precision_score(
            val_labels,
            val_probs
        )
    }


# ============================================================
# 9. TRAIN ONE MODEL
# ============================================================

train_positive = sum(
    patient_data[sid]["label"]
    for sid in train_ids
)

train_negative = (
    len(train_ids) - train_positive
)

POS_WEIGHT = (
    train_negative / train_positive
)

print("\nTraining positives:", train_positive)
print("Training negatives:", train_negative)
print(
    "Positive class weight:",
    round(POS_WEIGHT, 4)
)


def train_model(rnn_type):

    print(
        "\n"
        + "=" * 65
    )
    print(
        "Training",
        rnn_type,
        "365d baseline"
    )
    print("=" * 65)

    # Reset seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = RNNClassifier(
        VOCAB_SIZE,
        rnn_type
    ).to(DEVICE)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            POS_WEIGHT,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val_auc = -1
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            logits = model(
                batch["codes"].to(DEVICE),
                batch["length"]
            )

            labels = batch[
                "label"
            ].to(DEVICE)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

        _, val_labels, val_probs = predict(
            model,
            val_loader
        )

        val_auc = roc_auc_score(
            val_labels,
            val_probs
        )

        val_auprc = average_precision_score(
            val_labels,
            val_probs
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {running_loss / len(train_loader):.4f} | "
            f"Val AUROC {val_auc:.4f} | "
            f"Val AUPRC {val_auprc:.4f}"
        )

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= PATIENCE:

                print(
                    "Early stopping at epoch",
                    epoch
                )

                break

    model.load_state_dict(
        best_state
    )

    _, val_labels, val_probs = predict(
        model,
        val_loader
    )

    test_subjects, test_labels, test_probs = predict(
        model,
        test_loader
    )

    result = evaluate(
        f"{rnn_type} 365d",
        val_labels,
        val_probs,
        test_labels,
        test_probs
    )

    result[
        "Best_Val_AUROC"
    ] = best_val_auc

    result[
        "Best_Epoch"
    ] = best_epoch

    print(
        "\nBest validation AUROC:",
        round(best_val_auc, 6),
        "at epoch",
        best_epoch
    )

    print(
        "\n"
        + "=" * 65
    )
    print(
        f"LEAKAGE-SAFE {rnn_type} BASELINE RESULTS"
    )
    print("=" * 65)

    for key, value in result.items():

        if isinstance(
            value,
            (float, np.floating)
        ):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    checkpoint = (
        rnn_type.lower()
        + "_365d_best.pt"
    )

    torch.save(
        best_state,
        checkpoint
    )

    prediction_df = pd.DataFrame({
        "subject_id": test_subjects,
        "label": test_labels,
        "probability": test_probs
    })

    prediction_file = (
        rnn_type.lower()
        + "_365d_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    return result


# ============================================================
# 10. RUN LSTM + GRU
# ============================================================

results = []

results.append(
    train_model("LSTM")
)

results.append(
    train_model("GRU")
)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "lstm_gru_365d_results.csv",
    index=False
)

print(
    "\nSaved: lstm_gru_365d_results.csv"
)
print(
    "Saved: lstm_365d_predictions.csv"
)
print(
    "Saved: gru_365d_predictions.csv"
)
print(
    "Saved: lstm_365d_best.pt"
)
print(
    "Saved: gru_365d_best.pt"
)
