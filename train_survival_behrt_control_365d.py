import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

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

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
PATIENCE = 4

# 12 monthly survival intervals across 365 days
N_INTERVALS = 12
INTERVAL_DAYS = 365.0 / N_INTERVALS

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_timeaware_behrt_365d.parquet"
)

surv = pd.read_parquet(
    "cohort_survival_final_365d.parquet"
)

seq["subject_id"] = seq["subject_id"].astype(int)
surv["subject_id"] = surv["subject_id"].astype(int)

print("\nPatients:", len(surv))
print("DN events:", int(surv["dn_event"].sum()))
print(
    "Censored:",
    int((surv["dn_event"] == 0).sum())
)


# ============================================================
# 3. FIXED PATIENT SPLITS
# ============================================================

train_ids = surv.loc[
    surv["split"] == "train",
    "subject_id"
].tolist()

val_ids = surv.loc[
    surv["split"] == "validation",
    "subject_id"
].tolist()

test_ids = surv.loc[
    surv["split"] == "test",
    "subject_id"
].tolist()

print("\nTrain:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# 4. TRAINING VOCABULARY ONLY
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

for i, code in enumerate(top_codes, start=2):
    code_to_id[code] = i

VOCAB_SIZE = len(code_to_id)

print("\nVocabulary size:", VOCAB_SIZE)


# ============================================================
# 5. SURVIVAL LOOKUPS
# ============================================================

time_map = dict(
    zip(
        surv["subject_id"],
        surv["time_to_dn_days"]
    )
)

event_map = dict(
    zip(
        surv["subject_id"],
        surv["dn_event"]
    )
)


# ============================================================
# 6. BUILD PATIENT SEQUENCES
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

for sid, g in seq.groupby("subject_id"):

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

    patient_data[int(sid)] = {

        "codes": [
            code_to_id.get(x, 1)
            for x in g["icd_token"]
        ],

        "ages": (
            g["visit_age"]
            .fillna(0)
            .astype(int)
            .clip(0, 120)
            .tolist()
        ),

        "segments": (
            (
                (g["visit_number"].astype(int) - 1)
                % 2
            )
            .tolist()
        ),

        "gaps": (
            g["time_gap_bucket"]
            .astype(int)
            .clip(0, 8)
            .tolist()
        ),

        "recency": (
            g["time_to_cutoff_bucket"]
            .astype(int)
            .clip(0, 8)
            .tolist()
        )
    }

print(
    "Patients truncated to",
    MAX_LEN,
    "codes:",
    truncated
)


# ============================================================
# 7. DATASET
# ============================================================

class SurvivalDataset(Dataset):

    def __init__(self, patient_ids):
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        sid = int(self.patient_ids[idx])

        p = patient_data[sid]

        codes = p["codes"]
        ages = p["ages"]
        segments = p["segments"]
        gaps = p["gaps"]
        recency = p["recency"]

        length = len(codes)

        positions = list(range(length))

        pad = MAX_LEN - length

        codes += [0] * pad
        ages += [0] * pad
        segments += [0] * pad
        gaps += [0] * pad
        recency += [0] * pad
        positions += [0] * pad

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

            "time": torch.tensor(
                float(time_map[sid]),
                dtype=torch.float32
            ),

            "event": torch.tensor(
                float(event_map[sid]),
                dtype=torch.float32
            )
        }


train_loader = DataLoader(
    SurvivalDataset(train_ids),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    SurvivalDataset(val_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    SurvivalDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. TIME-AWARE BEHRT SURVIVAL MODEL
# ============================================================

class SurvivalBEHRT(nn.Module):

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

        # One hazard probability for each monthly interval
        self.survival_head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, N_INTERVALS)
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

        padding_mask = codes.eq(0)

        x = (
            self.code_embedding(codes)
            + self.age_embedding(ages)
            + self.segment_embedding(segments)
            + self.position_embedding(positions)

    )

        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        valid = (
            (~padding_mask)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            (x * valid).sum(dim=1)
            /
            valid.sum(dim=1).clamp(min=1.0)
        )

        return self.survival_head(pooled)


model = SurvivalBEHRT().to(DEVICE)

print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 9. DISCRETE-TIME SURVIVAL LOSS
# ============================================================

def survival_loss(logits, times, events):

    hazards = torch.sigmoid(logits)

    hazards = torch.clamp(
        hazards,
        min=1e-7,
        max=1 - 1e-7
    )

    event_bin = torch.floor(
        times / INTERVAL_DAYS
    ).long()

    event_bin = torch.clamp(
        event_bin,
        0,
        N_INTERVALS - 1
    )

    idx = torch.arange(
        N_INTERVALS,
        device=logits.device
    ).unsqueeze(0)

    bins = event_bin.unsqueeze(1)

    event_mask = (
        (idx == bins)
        &
        (events.unsqueeze(1) == 1)
    )

    survival_mask_event = (
        (idx < bins)
        &
        (events.unsqueeze(1) == 1)
    )

    survival_mask_censor = (
        (idx <= bins)
        &
        (events.unsqueeze(1) == 0)
    )

    survival_mask = (
        survival_mask_event
        |
        survival_mask_censor
    )

    log_likelihood = (
        survival_mask.float()
        * torch.log(1 - hazards)
    ).sum(dim=1)

    log_likelihood += (
        event_mask.float()
        * torch.log(hazards)
    ).sum(dim=1)

    return -log_likelihood.mean()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# 10. CONCORDANCE INDEX
# ============================================================

def concordance_index(
    times,
    events,
    risks
):

    concordant = 0.0
    comparable = 0.0

    n = len(times)

    for i in range(n):

        if events[i] != 1:
            continue

        mask = times > times[i]

        if not np.any(mask):
            continue

        other_risks = risks[mask]

        comparable += len(other_risks)

        concordant += np.sum(
            risks[i] > other_risks
        )

        concordant += 0.5 * np.sum(
            risks[i] == other_risks
        )

    if comparable == 0:
        return np.nan

    return concordant / comparable


# ============================================================
# 11. PREDICTION
# ============================================================

def predict(loader):

    model.eval()

    all_subjects = []
    all_times = []
    all_events = []
    all_risks = []

    with torch.no_grad():

        for batch in loader:

            logits = model(
                batch["codes"].to(DEVICE),
                batch["ages"].to(DEVICE),
                batch["segments"].to(DEVICE),
                batch["positions"].to(DEVICE),
                batch["gaps"].to(DEVICE),
                batch["recency"].to(DEVICE)
            )

            hazards = torch.sigmoid(logits)

            survival_365 = torch.prod(
                1 - hazards,
                dim=1
            )

            risk_365 = 1 - survival_365

            all_subjects.extend(
                batch["subject_id"]
                .numpy()
                .tolist()
            )

            all_times.extend(
                batch["time"]
                .numpy()
                .tolist()
            )

            all_events.extend(
                batch["event"]
                .numpy()
                .astype(int)
                .tolist()
            )

            all_risks.extend(
                risk_365
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(all_subjects),
        np.asarray(all_times),
        np.asarray(all_events),
        np.asarray(all_risks)
    )


# ============================================================
# 12. TRAIN
# ============================================================

BEST_MODEL = (
    "survival_behrt_control_365d_best.pt"
)

best_cindex = -np.inf
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for batch in train_loader:

        optimizer.zero_grad()

        logits = model(
            batch["codes"].to(DEVICE),
            batch["ages"].to(DEVICE),
            batch["segments"].to(DEVICE),
            batch["positions"].to(DEVICE),
            batch["gaps"].to(DEVICE),
            batch["recency"].to(DEVICE)
        )

        loss = survival_loss(
            logits,
            batch["time"].to(DEVICE),
            batch["event"].to(DEVICE)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += loss.item()

    _, val_time, val_event, val_risk = predict(
        val_loader
    )

    val_cindex = concordance_index(
        val_time,
        val_event,
        val_risk
    )

    val_auc = roc_auc_score(
        val_event,
        val_risk
    )

    mean_loss = (
        running_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {mean_loss:.4f} | "
        f"Val C-index {val_cindex:.4f} | "
        f"Val 365d AUROC {val_auc:.4f}"
    )

    if val_cindex > best_cindex:

        best_cindex = val_cindex
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping at epoch",
                epoch
            )

            break


print(
    "\nBest validation C-index:",
    round(best_cindex, 4),
    "at epoch",
    best_epoch
)


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)

test_subjects, test_time, test_event, test_risk = predict(
    test_loader
)

test_cindex = concordance_index(
    test_time,
    test_event,
    test_risk
)

test_auc = roc_auc_score(
    test_event,
    test_risk
)

test_auprc = average_precision_score(
    test_event,
    test_risk
)

results = {
    "Model":
        "BEHRT Survival Control 365d",

    "N_test":
        len(test_event),

    "Events_test":
        int(test_event.sum()),

    "C_index":
        test_cindex,

    "AUROC_365d":
        test_auc,

    "AUPRC_365d":
        test_auprc,

    "Best_Val_C_index":
        best_cindex,

    "Best_Epoch":
        best_epoch
}


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TIME-TO-FIRST-DOCUMENTED-DN SURVIVAL RESULTS")
print("=" * 60)

for key, value in results.items():

    if isinstance(value, float):
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
    "survival_behrt_control_365d_results.csv",
    index=False
)

pd.DataFrame({
    "subject_id": test_subjects,
    "time_to_dn_days": test_time,
    "dn_event": test_event,
    "predicted_365d_risk": test_risk
}).to_csv(
    "survival_behrt_control_365d_predictions.csv",
    index=False
)

print(
    "\nSaved: survival_behrt_control_365d_results.csv"
)

print(
    "Saved: survival_behrt_control_365d_predictions.csv"
)

print(
    "Saved:",
    BEST_MODEL
)
