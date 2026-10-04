import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = int(os.environ.get("SURVIVAL_SEED", "42"))

MODEL_PATH = "./Bio_ClinicalBERT"

MAX_LENGTH = 256
BATCH_SIZE = 16

EPOCHS = 8
LR = 2e-5
PATIENCE = 2

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
# 2. LOAD LEAKAGE-SAFE DATA
# ============================================================

seq = pd.read_parquet(
    "dx_sequences_biobert_365d.parquet"
)

surv = pd.read_parquet(
    "cohort_survival_final_365d.parquet"
)

seq["subject_id"] = seq["subject_id"].astype(int)
surv["subject_id"] = surv["subject_id"].astype(int)

print("\nPatients:", len(surv))
print(
    "DN events:",
    int(surv["dn_event"].sum())
)
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
# 4. SURVIVAL LOOKUPS
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
# 5. BUILD SAME NON-TIME-AWARE CLINICALBERT TEXT
# ============================================================

seq = seq.sort_values(
    [
        "subject_id",
        "visit_number",
        "code_rank"
    ]
)

patient_text = {}

for sid, patient_df in seq.groupby("subject_id"):

    visit_texts = []

    for visit_number, visit_df in patient_df.groupby(
        "visit_number",
        sort=True
    ):

        visit_df = visit_df.sort_values(
            "code_rank"
        )

        codes = (
            visit_df["diagnosis_text"]
            .astype(str)
            .tolist()
        )

        visit_text = (
            "Visit "
            + str(int(visit_number))
            + " diagnoses: "
            + "; ".join(codes)
            + "."
        )

        visit_texts.append(
            visit_text
        )

    text = (
        "Patient prior diagnosis history. "
        + " ".join(visit_texts)
    )

    patient_text[int(sid)] = text


print(
    "\nPatient texts created:",
    len(patient_text)
)

missing = [
    sid for sid in surv["subject_id"]
    if sid not in patient_text
]

print(
    "Patients missing text:",
    len(missing)
)

if len(missing) > 0:
    raise ValueError(
        "Some survival patients are missing ClinicalBERT text."
    )


# ============================================================
# 6. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print(
    "\nClinicalBERT tokenizer loaded from:",
    MODEL_PATH
)


# ============================================================
# 7. DATASET
# ============================================================

class ClinicalBERTSurvivalDataset(Dataset):

    def __init__(self, patient_ids):
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        sid = int(
            self.patient_ids[idx]
        )

        encoded = tokenizer(
            patient_text[sid],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "subject_id":
                torch.tensor(
                    sid,
                    dtype=torch.long
                ),

            "input_ids":
                encoded[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoded[
                    "attention_mask"
                ].squeeze(0),

            "time":
                torch.tensor(
                    float(time_map[sid]),
                    dtype=torch.float32
                ),

            "event":
                torch.tensor(
                    float(event_map[sid]),
                    dtype=torch.float32
                )
        }


train_loader = DataLoader(
    ClinicalBERTSurvivalDataset(
        train_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    ClinicalBERTSurvivalDataset(
        val_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    ClinicalBERTSurvivalDataset(
        test_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. CLINICALBERT SURVIVAL MODEL
# ============================================================

class ClinicalBERTSurvival(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        hidden_size = (
            self.bert.config.hidden_size
        )

        # Same general survival-head structure
        # as the BEHRT survival model.
        self.survival_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_size,
                N_INTERVALS
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if (
            hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            pooled = outputs.pooler_output
        else:
            pooled = (
                outputs
                .last_hidden_state[:, 0, :]
            )

        return self.survival_head(
            pooled
        )


model = ClinicalBERTSurvival().to(
    DEVICE
)

print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 9. DISCRETE-TIME SURVIVAL LOSS
# ============================================================

def survival_loss(
    logits,
    times,
    events
):

    hazards = torch.sigmoid(
        logits
    )

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
        * torch.log(
            1 - hazards
        )
    ).sum(dim=1)

    log_likelihood += (
        event_mask.float()
        * torch.log(
            hazards
        )
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

        comparable += len(
            other_risks
        )

        concordant += np.sum(
            risks[i] > other_risks
        )

        concordant += (
            0.5
            * np.sum(
                risks[i]
                == other_risks
            )
        )

    if comparable == 0:
        return np.nan

    return (
        concordant
        / comparable
    )


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
                input_ids=
                    batch[
                        "input_ids"
                    ].to(DEVICE),

                attention_mask=
                    batch[
                        "attention_mask"
                    ].to(DEVICE)
            )

            hazards = torch.sigmoid(
                logits
            )

            survival_365 = torch.prod(
                1 - hazards,
                dim=1
            )

            risk_365 = (
                1 - survival_365
            )

            all_subjects.extend(
                batch[
                    "subject_id"
                ]
                .numpy()
                .tolist()
            )

            all_times.extend(
                batch[
                    "time"
                ]
                .numpy()
                .tolist()
            )

            all_events.extend(
                batch[
                    "event"
                ]
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
        np.asarray(
            all_subjects
        ),

        np.asarray(
            all_times
        ),

        np.asarray(
            all_events
        ),

        np.asarray(
            all_risks
        )
    )


# ============================================================
# 12. TRAIN
# ============================================================

BEST_MODEL = f"survival_clinicalbert_365d_best_seed{SEED}.pt"
RESULTS_FILE = f"survival_clinicalbert_365d_results_seed{SEED}.csv"
PREDICTIONS_FILE = f"survival_clinicalbert_365d_predictions_seed{SEED}.csv"

best_cindex = -np.inf
best_epoch = 0
patience_counter = 0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for batch in train_loader:

        optimizer.zero_grad()

        logits = model(
            input_ids=
                batch[
                    "input_ids"
                ].to(DEVICE),

            attention_mask=
                batch[
                    "attention_mask"
                ].to(DEVICE)
        )

        loss = survival_loss(
            logits,
            batch[
                "time"
            ].to(DEVICE),
            batch[
                "event"
            ].to(DEVICE)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

    (
        _,
        val_time,
        val_event,
        val_risk
    ) = predict(
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

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping at epoch",
                epoch
            )

            break


print(
    "\nBest validation C-index:",
    round(
        best_cindex,
        4
    ),
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

(
    test_subjects,
    test_time,
    test_event,
    test_risk
) = predict(
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
        "ClinicalBERT Survival 365d",

    "Seed":
        SEED,

    "N_test":
        len(test_event),

    "Events_test":
        int(
            test_event.sum()
        ),

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

print(
    "CLINICALBERT TIME-TO-FIRST-DOCUMENTED-DN "
    "SURVIVAL RESULTS"
)

print("=" * 60)

for key, value in results.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key}: "
            f"{value:.6f}"
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
    RESULTS_FILE,
    index=False
)

pd.DataFrame(
    {
        "subject_id":
            test_subjects,

        "time_to_dn_days":
            test_time,

        "dn_event":
            test_event,

        "predicted_365d_risk":
            test_risk
    }
).to_csv(
    PREDICTIONS_FILE,
    index=False
)

print("\nSaved:", RESULTS_FILE)
print("Saved:", PREDICTIONS_FILE)

print(
    "Saved:",
    BEST_MODEL
)
