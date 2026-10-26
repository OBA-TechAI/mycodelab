
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ============================================================
# 1. SETTINGS
# ============================================================
SEQ_FILE = "dx_sequences_biobert_365d.parquet"

MODEL_PATH = "./Llama-3.2-3B-Instruct"
ADAPTER_PATH = (
    "./llama32_seed42_frozen/"
    "llama32_3b_lora_timeaware_365d_adapter"
)

PRED_FILE = (
    "llama32_seed42_frozen/"
    "llama32_3b_lora_timeaware_365d_predictions.csv"
)

OUTPUT_DIR = "rq6_llama_timeaware_outputs"

MAX_LENGTH = 1024
BATCH_SIZE = 2
TOP_N = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")

print("Device: cuda", flush=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

seq = pd.read_parquet(SEQ_FILE).sort_values(
    ["subject_id", "visit_number", "code_rank"]
)

assert (
    pd.to_datetime(seq["admittime"])
    < pd.to_datetime(seq["prediction_cutoff"])
).all()

seq["subject_id"] = seq["subject_id"].astype(int)

original = pd.read_csv(PRED_FILE)
original["subject_id"] = original["subject_id"].astype(int)

test_ids = original["subject_id"].tolist()
test_set = set(test_ids)

test_seq = seq[seq["subject_id"].isin(test_set)].copy()

assert test_seq["subject_id"].nunique() == len(test_ids)
assert len(test_ids) == 1337
assert original["subject_id"].is_unique

labels = original["label"].to_numpy(dtype=int)

print("Test patients:", len(test_ids), flush=True)

# ============================================================
# 3. RECONSTRUCT TIME-AWARE PATIENT TEXT
# ============================================================

patient_visits = {}

for sid, patient_df in test_seq.groupby("subject_id"):
    visits = []

    for visit_number, visit_df in patient_df.groupby(
        "visit_number", sort=True
    ):
        visit_df = visit_df.sort_values("code_rank")

        descriptions = (
            visit_df["diagnosis_text"]
            .astype(str)
            .tolist()
        )

        recency = float(
            visit_df["time_to_cutoff_days"].iloc[0]
        )

        gap = float(
            visit_df["time_gap_days"].iloc[0]
        )

        visits.append((
            int(visit_number),
            descriptions,
            recency,
            gap
        ))

    patient_visits[int(sid)] = visits


def build_text(sid, target=None, temporal_mask=None):
    visit_texts = []

    for visit_number, descriptions, recency, gap in patient_visits[sid]:

        if target is None:
            selected = descriptions
        else:
            selected = [
                "diagnosis information masked"
                if description == target else description
                for description in descriptions
            ]

        if temporal_mask == "recency_days":
            recency = 0.0

        if temporal_mask == "visit_gap_days":
            gap = 0.0

        visit_text = (
            "Visit "
            + str(visit_number)
            + ", "
            + str(round(recency, 1))
            + " days before prediction cutoff, "
            + str(round(gap, 1))
            + " days since previous visit. "
            + "Diagnoses: "
            + "; ".join(selected)
            + "."
        )

        visit_texts.append(visit_text)

    return (
        "Patient prior diagnosis history. "
        + " ".join(visit_texts)
    )

# ============================================================
# 4. LOAD LLAMA AND SAVED LoRA ADAPTER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH, local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
    low_cpu_mem_usage=True
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    is_trainable=False
)

model.eval()
print("LoRA adapter loaded.", flush=True)

# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict(target=None, temporal_mask=None):
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(test_ids), BATCH_SIZE):
            ids = test_ids[start:start + BATCH_SIZE]
            texts = [
	       build_text(sid, target, temporal_mask)
    	       for sid in ids
	    ]

            encoded = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                k: v.to("cuda")
                for k, v in encoded.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                outputs = model(**encoded)
                logits = outputs.logits.squeeze(-1).float()

            probabilities.extend(
                torch.sigmoid(logits).cpu().numpy().tolist()
            )

    return np.asarray(probabilities, dtype=float)


def metrics(probs):
    return (
        roc_auc_score(labels, probs),
        average_precision_score(labels, probs)
    )

# ============================================================
# 6. BASELINE VERIFICATION
# ============================================================

baseline_probs = predict()

baseline_auc, baseline_auprc = metrics(baseline_probs)

max_diff = np.max(np.abs(
    baseline_probs
    - original["probability"].to_numpy(dtype=float)
))

print("Reconstructed baseline AUROC:", baseline_auc, flush=True)
print("Reconstructed baseline AUPRC:", baseline_auprc, flush=True)
print("Max absolute prediction difference:", max_diff, flush=True)

pd.DataFrame({
    "subject_id": test_ids,
    "label": labels,
    "probability": baseline_probs
}).to_csv(
    os.path.join(OUTPUT_DIR, "baseline_predictions.csv"),
    index=False
)

if max_diff > 1e-4:
    raise RuntimeError(
        "BASELINE VERIFICATION FAILED. "
        "Do not interpret masking results."
    )

print("BASELINE VERIFIED", flush=True)

# ============================================================
# 7. SELECT 30 FREQUENT DIAGNOSIS DESCRIPTIONS
# ============================================================

counts = Counter(
    test_seq["diagnosis_text"].astype(str).tolist()
)

top_descriptions = [
    description
    for description, count in counts.most_common(TOP_N)
]

print(
    "Diagnosis descriptions selected:",
    len(top_descriptions),
    flush=True
)

# ============================================================
# 8. DIAGNOSIS-DESCRIPTION MASKING
# ============================================================

results = []

for index, description in enumerate(top_descriptions, start=1):
    perturbed_probs = predict(target=description)
    auc, auprc = metrics(perturbed_probs)

    auc_drop = baseline_auc - auc
    auprc_drop = baseline_auprc - auprc

    results.append({
        "diagnosis_description": description,
        "test_frequency": counts[description],
        "baseline_auroc": baseline_auc,
        "masked_auroc": auc,
        "auroc_drop": auc_drop,
        "baseline_auprc": baseline_auprc,
        "masked_auprc": auprc,
        "auprc_drop": auprc_drop
    })

    print(
        index,
        description[:75],
        "AUROC drop:",
        round(auc_drop, 6),
        "AUPRC drop:",
        round(auprc_drop, 6),
        flush=True
    )

pd.DataFrame(results).sort_values(
    "auroc_drop", ascending=False
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "diagnosis_text_masking_importance.csv"
    ),
    index=False
)

pd.DataFrame([{
    "model": "Llama 3.2-3B LoRA Control",
    "test_patients": len(test_ids),
    "auroc": baseline_auc,
    "auprc": baseline_auprc,
    "max_prediction_difference": max_diff
}]).to_csv(
    os.path.join(OUTPUT_DIR, "baseline_metrics.csv"),
    index=False
)

# ============================================================
# 9. TEMPORAL-FEATURE IMPORTANCE
# ============================================================

temporal_results = []

for feature in ["recency_days", "visit_gap_days"]:
    perturbed_probs = predict(temporal_mask=feature)
    auc, auprc = metrics(perturbed_probs)

    auc_drop = baseline_auc - auc
    auprc_drop = baseline_auprc - auprc

    temporal_results.append({
        "feature": feature,
        "baseline_auroc": baseline_auc,
        "masked_auroc": auc,
        "auroc_drop": auc_drop,
        "baseline_auprc": baseline_auprc,
        "masked_auprc": auprc,
        "auprc_drop": auprc_drop
    })

    print(
        "Temporal feature:", feature,
        "AUROC drop:", round(auc_drop, 6),
        "AUPRC drop:", round(auprc_drop, 6),
        flush=True
    )

pd.DataFrame(temporal_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "temporal_text_masking_importance.csv"
    ),
    index=False
)

print("RQ6 TIME-AWARE LLAMA COMPLETED", flush=True)
print("Saved outputs:", os.path.abspath(OUTPUT_DIR), flush=True)

