import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

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

MODEL_PATH = "./Clinical-Longformer"

MAX_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

EPOCHS = 6
LR = 1e-5
PATIENCE = 2

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

cohort = pd.read_parquet(
    "cohort_labels_timeaware_365d.parquet"
)[[
    "subject_id",
    "dn_within_365d"
]]

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

cohort = cohort.merge(
    splits,
    on="subject_id",
    how="inner"
)

seq["subject_id"] = (
    seq["subject_id"].astype(int)
)

cohort["subject_id"] = (
    cohort["subject_id"].astype(int)
)

print("\nPatients:", len(cohort))

print(
    "DN positive:",
    int(
        cohort[
            "dn_within_365d"
        ].sum()
    )
)

print(
    "DN negative:",
    int(
        (
            cohort[
                "dn_within_365d"
            ] == 0
        ).sum()
    )
)


# ============================================================
# 3. FIXED SPLITS
# ============================================================

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

print("\nTrain:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# 4. LABEL LOOKUP
# ============================================================

label_map = dict(
    zip(
        cohort["subject_id"],
        cohort["dn_within_365d"]
    )
)


# ============================================================
# 5. BUILD NON-TIME-AWARE CLINICAL TEXT
# ============================================================

seq = seq.sort_values(
    [
        "subject_id",
        "visit_number",
        "code_rank"
    ]
)

patient_text = {}

for sid, patient_df in seq.groupby(
    "subject_id"
):

    visit_texts = []

    for visit_number, visit_df in (
        patient_df.groupby(
            "visit_number",
            sort=True
        )
    ):

        visit_df = visit_df.sort_values(
            "code_rank"
        )

        diagnoses = (
            visit_df[
                "diagnosis_text"
            ]
            .astype(str)
            .tolist()
        )

        visit_text = (
            "Visit "
            + str(
                int(visit_number)
            )
            + " diagnoses: "
            + "; ".join(
                diagnoses
            )
            + "."
        )

        visit_texts.append(
            visit_text
        )

    text = (
        "Patient prior diagnosis history. "
        + " ".join(
            visit_texts
        )
    )

    patient_text[
        int(sid)
    ] = text


print(
    "Patient texts created:",
    len(patient_text)
)

example_id = train_ids[0]

print("\nExample text:")

print(
    patient_text[
        example_id
    ][:600]
)


# ============================================================
# 6. LOAD CLINICAL LONGFORMER LOCALLY
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH,
        num_labels=1,
        local_files_only=True
    )
)

# Reduce GPU memory use
model.gradient_checkpointing_enable()

model = model.to(DEVICE)

print(
    "\nClinical Longformer loaded from:",
    MODEL_PATH
)

print(
    "Trainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 7. DATASET
# ============================================================

class LongformerDataset(
    Dataset
):

    def __init__(
        self,
        patient_ids
    ):

        self.patient_ids = (
            patient_ids
        )

    def __len__(self):

        return len(
            self.patient_ids
        )

    def __getitem__(
        self,
        idx
    ):

        sid = int(
            self.patient_ids[
                idx
            ]
        )

        text = patient_text[
            sid
        ]

        encoded = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = (
            encoded[
                "input_ids"
            ].squeeze(0)
        )

        attention_mask = (
            encoded[
                "attention_mask"
            ].squeeze(0)
        )

        # Longformer global attention
        # on the first token
        global_attention_mask = (
            torch.zeros_like(
                input_ids
            )
        )

        global_attention_mask[
            0
        ] = 1

        return {

            "subject_id":
                sid,

            "input_ids":
                input_ids,

            "attention_mask":
                attention_mask,

            "global_attention_mask":
                global_attention_mask,

            "label":
                torch.tensor(
                    float(
                        label_map[
                            sid
                        ]
                    ),
                    dtype=torch.float32
                )
        }


train_loader = DataLoader(
    LongformerDataset(
        train_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    LongformerDataset(
        val_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    LongformerDataset(
        test_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. CLASS WEIGHT
# ============================================================

train_labels = cohort[
    cohort["split"] == "train"
]["dn_within_365d"]

n_pos = int(
    train_labels.sum()
)

n_neg = int(
    (
        train_labels == 0
    ).sum()
)

pos_weight_value = (
    n_neg / n_pos
)

print(
    "\nTraining positives:",
    n_pos
)

print(
    "Training negatives:",
    n_neg
)

print(
    "Positive class weight:",
    round(
        pos_weight_value,
        4
    )
)

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

criterion = (
    nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# 9. PREDICTION
# ============================================================

def predict(loader):

    model.eval()

    subject_ids = []
    labels = []
    probabilities = []

    with torch.no_grad():

        for batch in loader:

            outputs = model(

                input_ids=
                    batch[
                        "input_ids"
                    ].to(DEVICE),

                attention_mask=
                    batch[
                        "attention_mask"
                    ].to(DEVICE),

                global_attention_mask=
                    batch[
                        "global_attention_mask"
                    ].to(DEVICE)
            )

            logits = (
                outputs.logits
                .squeeze(-1)
            )

            probs = torch.sigmoid(
                logits
            )

            subject_ids.extend(
                batch[
                    "subject_id"
                ].cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch[
                    "label"
                ].cpu()
                .numpy()
                .tolist()
            )

            probabilities.extend(
                probs.cpu()
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
# 10. TRAIN
# ============================================================

BEST_MODEL = (
    "clinical_longformer_365d_control_best.pt"
)

best_auc = -np.inf
best_epoch = 0
patience_counter = 0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad()

    running_loss = 0.0

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        outputs = model(

            input_ids=
                batch[
                    "input_ids"
                ].to(DEVICE),

            attention_mask=
                batch[
                    "attention_mask"
                ].to(DEVICE),

            global_attention_mask=
                batch[
                    "global_attention_mask"
                ].to(DEVICE)
        )

        logits = (
            outputs.logits
            .squeeze(-1)
        )

        labels = (
            batch[
                "label"
            ].to(DEVICE)
        )

        loss = criterion(
            logits,
            labels
        )

        loss = (
            loss
            / GRAD_ACCUM_STEPS
        )

        loss.backward()

        running_loss += (
            loss.item()
            * GRAD_ACCUM_STEPS
        )

        if (
            step
            % GRAD_ACCUM_STEPS
            == 0
            or
            step
            == len(train_loader)
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            optimizer.zero_grad()


    _, val_y, val_prob = (
        predict(
            val_loader
        )
    )

    val_auc = roc_auc_score(
        val_y,
        val_prob
    )

    val_auprc = (
        average_precision_score(
            val_y,
            val_prob
        )
    )

    mean_loss = (
        running_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {mean_loss:.4f} | "
        f"Val AUROC {val_auc:.4f} | "
        f"Val AUPRC {val_auprc:.4f}"
    )

    if val_auc > best_auc:

        best_auc = (
            val_auc
        )

        best_epoch = (
            epoch
        )

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
    "\nBest validation AUROC:",
    round(
        best_auc,
        4
    ),
    "at epoch",
    best_epoch
)


# ============================================================
# 11. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)


# ============================================================
# 12. VALIDATION THRESHOLD
# ============================================================

_, val_y, val_prob = (
    predict(
        val_loader
    )
)

best_threshold = 0.50
best_f1 = -1

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

        best_f1 = (
            score
        )

        best_threshold = float(
            threshold
        )


print(
    "Selected validation threshold:",
    round(
        best_threshold,
        3
    )
)


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

test_subjects, test_y, test_prob = (
    predict(
        test_loader
    )
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
    ).ravel()
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

results = {

    "Model":
        "Clinical Longformer Diagnosis-Text 365d Control",

    "N_test":
        len(test_y),

    "AUROC":
        roc_auc_score(
            test_y,
            test_prob
        ),

    "AUPRC":
        average_precision_score(
            test_y,
            test_prob
        ),

    "Accuracy":
        accuracy_score(
            test_y,
            test_pred
        ),

    "Precision":
        precision_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Recall_Sensitivity":
        recall_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Specificity":
        specificity,

    "F1":
        f1_score(
            test_y,
            test_pred,
            zero_division=0
        ),

    "Brier":
        brier_score_loss(
            test_y,
            test_prob
        ),

    "Threshold":
        best_threshold,

    "Best_Val_AUROC":
        best_auc,

    "Best_Epoch":
        best_epoch
}


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)

print(
    "LEAKAGE-SAFE CLINICAL LONGFORMER CONTROL RESULTS"
)

print("=" * 70)

for key, value in (
    results.items()
):

    if isinstance(
        value,
        float
    ):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# 15. SAVE
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    "clinical_longformer_365d_control_results.csv",
    index=False
)

pd.DataFrame({

    "subject_id":
        test_subjects,

    "y_true":
        test_y,

    "y_prob":
        test_prob,

    "y_pred":
        test_pred

}).to_csv(
    "clinical_longformer_365d_control_predictions.csv",
    index=False
)

print(
    "\nSaved: clinical_longformer_365d_control_results.csv"
)

print(
    "Saved: clinical_longformer_365d_control_predictions.csv"
)

print(
    "Saved:",
    BEST_MODEL
)
