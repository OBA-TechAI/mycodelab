import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
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

MODEL_PATH = "./Llama-3.2-3B-Instruct"
SEQ_FILE = "dx_sequences_biobert_365d.parquet"

SEED = 42
MAX_LENGTH = 1024

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

MAX_EPOCHS = 5
PATIENCE = 2
LR = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BEST_DIR = "llama32_3b_lora_365d_control_adapter"

RESULT_FILE = "llama32_3b_lora_365d_control_results.csv"
PRED_FILE = "llama32_3b_lora_365d_control_predictions.csv"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if not torch.cuda.is_available():
    raise RuntimeError("GPU is required for Llama training")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. LOAD LEAKAGE-SAFE DATA
# ============================================================

seq = pd.read_parquet(
    SEQ_FILE
)

seq = seq.sort_values(
    [
        "subject_id",
        "visit_number",
        "code_rank"
    ]
)

print("\nPatients:", seq["subject_id"].nunique())
print(
    "DN positive:",
    seq.groupby("subject_id")[
        "dn_within_365d"
    ].first().sum()
)

print(
    "DN negative:",
    seq["subject_id"].nunique()
    - seq.groupby("subject_id")[
        "dn_within_365d"
    ].first().sum()
)

assert (
    pd.to_datetime(seq["admittime"])
    < pd.to_datetime(seq["prediction_cutoff"])
).all()

print("All historical events before cutoff: True")


# ============================================================
# 3. BUILD NON-TIME-AWARE DIAGNOSIS TEXT
# ============================================================

patient_text = {}
patient_label = {}
patient_split = {}

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
            visit_df["diagnosis_text"]
            .astype(str)
            .tolist()
        )

        visit_text = (
            "Visit "
            + str(int(visit_number))
            + " diagnoses: "
            + "; ".join(diagnoses)
            + "."
        )

        visit_texts.append(
            visit_text
        )

    text = (
        "Patient prior diagnosis history. "
        + " ".join(visit_texts)
    )

    sid = int(sid)

    patient_text[sid] = text

    patient_label[sid] = int(
        patient_df[
            "dn_within_365d"
        ].iloc[0]
    )

    patient_split[sid] = str(
        patient_df["split"].iloc[0]
    )


train_ids = [
    sid for sid in patient_text
    if patient_split[sid] == "train"
]

val_ids = [
    sid for sid in patient_text
    if patient_split[sid] == "validation"
]

test_ids = [
    sid for sid in patient_text
    if patient_split[sid] == "test"
]

print("\nTrain:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))

print(
    "Training positives:",
    sum(patient_label[sid] for sid in train_ids)
)

print(
    "\nExample text:"
)

print(
    patient_text[train_ids[0]][:1000]
)


# ============================================================
# 4. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(
    "\nTokenizer loaded from:",
    MODEL_PATH
)

print(
    "Maximum sequence length:",
    MAX_LENGTH
)


# ============================================================
# 5. DATASET
# ============================================================

class LlamaDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        sid = self.ids[idx]

        encoded = tokenizer(
            patient_text[sid],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "subject_id": torch.tensor(
                sid,
                dtype=torch.long
            ),

            "input_ids": encoded[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),

            "label": torch.tensor(
                patient_label[sid],
                dtype=torch.float32
            )
        }


train_loader = DataLoader(
    LlamaDataset(train_ids),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    LlamaDataset(val_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    LlamaDataset(test_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# 6. LOAD LLAMA CLASSIFIER
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
    low_cpu_mem_usage=True
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)

model.config.use_cache = False

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# ============================================================
# 7. LoRA
# ============================================================

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    modules_to_save=["score"],
    bias="none"
)

model = get_peft_model(
    model,
    lora_config
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nTotal parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Trainable percentage:",
    round(
        100 * trainable_params
        / total_params,
        4
    ),
    "%"
)


# ============================================================
# 8. CLASS WEIGHT
# ============================================================

train_positive = sum(
    patient_label[sid]
    for sid in train_ids
)

train_negative = (
    len(train_ids)
    - train_positive
)

POS_WEIGHT = (
    train_negative
    / train_positive
)

print(
    "\nTraining positives:",
    train_positive
)

print(
    "Training negatives:",
    train_negative
)

print(
    "Positive class weight:",
    round(POS_WEIGHT, 4)
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        POS_WEIGHT,
        dtype=torch.float32,
        device=DEVICE
    )
)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LR
)




# ============================================================
# 10. PREDICTION
# ============================================================

def predict(loader):

    model.eval()

    subject_ids = []
    labels = []
    probabilities = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(
                DEVICE,
                non_blocking=True
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(device_type="cuda",
                dtype=torch.bfloat16
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = (
                    outputs.logits
                    .squeeze(-1)
                    .float()
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
# 11. TRAINING
# ============================================================

best_val_auc = -1
best_epoch = 0
best_state = None
bad_epochs = 0

print("\nStarting Llama LoRA training...")

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(device_type="cuda",
            dtype=torch.bfloat16
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = (
                outputs.logits
                .squeeze(-1)
                .float()
            )

            loss = criterion(
                logits,
                labels
            )

            group_start = (
                (step - 1) // GRAD_ACCUM_STEPS
            ) * GRAD_ACCUM_STEPS

            accumulation_steps = min(
                GRAD_ACCUM_STEPS,
                len(train_loader) - group_start
            )

            loss = loss / accumulation_steps

        loss.backward()

        if (
            step % GRAD_ACCUM_STEPS == 0
            or step == len(train_loader)
        ):

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += (
            loss.item()
            * GRAD_ACCUM_STEPS
        )

    _, val_labels, val_probs = predict(
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
        f"Loss "
        f"{running_loss / len(train_loader):.4f} | "
        f"Val AUROC {val_auc:.4f} | "
        f"Val AUPRC {val_auprc:.4f}"
    )

    if val_auc > best_val_auc:

        best_val_auc = val_auc
        best_epoch = epoch

        state = get_peft_model_state_dict(
            model
        )

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in state.items()
        }

        bad_epochs = 0

    else:

        bad_epochs += 1

        if bad_epochs >= PATIENCE:

            print(
                "\nEarly stopping at epoch",
                epoch
            )

            break

# ============================================================
# SAFEGUARD BEFORE RESTORING BEST MODEL
# ============================================================

if best_state is None:
    raise RuntimeError(
        "No valid best model was saved. "
        "Check validation metrics and training logs."
    )

# ============================================================
# 12. RESTORE BEST MODEL
# ============================================================

set_peft_model_state_dict(
    model,
    best_state
)

print(
    "\nBest validation AUROC:",
    round(best_val_auc, 6),
    "at epoch",
    best_epoch
)


# ============================================================
# 13. VALIDATION THRESHOLD
# ============================================================

_, val_labels, val_probs = predict(
    val_loader
)

best_threshold = 0.50
best_f1 = -1

for threshold in np.arange(
    0.05,
    0.951,
    0.005
):

    preds = (
        val_probs >= threshold
    ).astype(int)

    score = f1_score(
        val_labels,
        preds,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(
            threshold
        )

print(
    "Selected validation threshold:",
    round(best_threshold, 3)
)


# ============================================================
# 14. TEST EVALUATION
# ============================================================

test_subjects, test_labels, test_probs = predict(
    test_loader
)

test_preds = (
    test_probs >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_labels,
    test_preds,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

results = {
    "Model":
        "Llama 3.2 3B Instruct LoRA Diagnosis-Text 365d Control",

    "N_test":
        len(test_labels),

    "AUROC":
        roc_auc_score(
            test_labels,
            test_probs
        ),

    "AUPRC":
        average_precision_score(
            test_labels,
            test_probs
        ),

    "Accuracy":
        accuracy_score(
            test_labels,
            test_preds
        ),

    "Precision":
        precision_score(
            test_labels,
            test_preds,
            zero_division=0
        ),

    "Recall_Sensitivity":
        recall_score(
            test_labels,
            test_preds,
            zero_division=0
        ),

    "Specificity":
        specificity,

    "F1":
        f1_score(
            test_labels,
            test_preds,
            zero_division=0
        ),

    "Brier":
        brier_score_loss(
            test_labels,
            test_probs
        ),

    "Threshold":
        best_threshold,

    "Best_Val_AUROC":
        best_val_auc,

    "Best_Epoch":
        best_epoch
}


print(
    "\n"
    + "=" * 72
)

print(
    "LEAKAGE-SAFE LLAMA 3.2 3B LoRA CONTROL RESULTS"
)

print(
    "=" * 72
)

for key, value in results.items():

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


# ============================================================
# 15. SAVE
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    RESULT_FILE,
    index=False
)

pd.DataFrame({
    "subject_id": test_subjects,
    "label": test_labels,
    "probability": test_probs
}).to_csv(
    PRED_FILE,
    index=False
)

model.save_pretrained(
    BEST_DIR
)

tokenizer.save_pretrained(
    BEST_DIR
)

print(
    "\nSaved:",
    RESULT_FILE
)

print(
    "Saved:",
    PRED_FILE
)

print(
    "Saved LoRA adapter:",
    BEST_DIR
)

print(
    "Peak GPU memory:",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2
    ),
    "GB"
)
