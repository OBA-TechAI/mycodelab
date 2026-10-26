
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
N_REPEATS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path("rq6_mlp_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "prior_dx_count",
    "prior_unique_dx_count",
    "prior_visit_count",
    "age_at_last_prior_visit",
    "mean_codes_per_visit",
    "max_codes_per_visit",
    "mean_unique_codes_per_visit",
    "prior_icd9_count",
    "prior_icd10_count"
]

TARGET = "dn_within_365d"


# ============================================================
# 2. LOAD ORIGINAL DATA AND SPLITS
# ============================================================

df = pd.read_parquet(
    "structured_features_365d_safe.parquet"
)

splits = pd.read_parquet(
    "patient_splits_timeaware_365d.parquet"
)

df = df.merge(
    splits,
    on="subject_id",
    how="inner"
)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = train_df[FEATURES].astype(float).values
X_test = test_df[FEATURES].astype(float).values

y_test = test_df[TARGET].astype(int).values
test_ids = test_df["subject_id"].values

scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

print("Test patients:", len(y_test), flush=True)
print("DN events:", int(y_test.sum()), flush=True)


# ============================================================
# 3. ORIGINAL MLP ARCHITECTURE
# ============================================================

class MLP(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


model = MLP(len(FEATURES)).to(DEVICE)

checkpoint = torch.load(
    "mlp_365d_best.pt",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Saved MLP checkpoint loaded.", flush=True)


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

@torch.no_grad()
def predict_probabilities(X):

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    logits = model(tensor)

    return torch.sigmoid(
        logits
    ).cpu().numpy()


# ============================================================
# 5. BASELINE VERIFICATION
# ============================================================

baseline_prob = predict_probabilities(
    X_test_scaled
)

baseline_auc = roc_auc_score(
    y_test, baseline_prob
)

baseline_ap = average_precision_score(
    y_test, baseline_prob
)

print("Reconstructed AUROC:", baseline_auc)
print("Reconstructed AUPRC:", baseline_ap)

original = pd.read_csv(
    "structured_baselines_365d_predictions.csv"
)

reconstructed = pd.DataFrame({
    "subject_id": test_ids,
    "label": y_test,
    "mlp_probability": baseline_prob
})

comparison = original[
    ["subject_id", "label", "mlp_probability"]
].merge(
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
    comparison["mlp_probability_original"].to_numpy()
    - comparison["mlp_probability_reconstructed"].to_numpy()
)

max_diff = float(differences.max())
mean_diff = float(differences.mean())

print("Maximum prediction difference:", max_diff)
print("Mean prediction difference:", mean_diff)

reconstructed.to_csv(
    OUTPUT_DIR / "rq6_mlp_reconstructed_predictions.csv",
    index=False
)

verified = max_diff <= 1e-5

pd.DataFrame([{
    "model": "MLP",
    "n_test": len(y_test),
    "dn_events": int(y_test.sum()),
    "auroc": baseline_auc,
    "auprc": baseline_ap,
    "max_prediction_difference": max_diff,
    "mean_prediction_difference": mean_diff,
    "verified": verified
}]).to_csv(
    OUTPUT_DIR / "rq6_mlp_baseline_verification.csv",
    index=False
)

if not verified:
    raise RuntimeError(
        "MLP baseline does not reproduce original predictions. "
        "Permutation analysis stopped."
    )

print("MLP BASELINE VERIFIED", flush=True)


# ============================================================
# 6. PERMUTATION FEATURE IMPORTANCE
# ============================================================

results = []

for feature_index, feature_name in enumerate(FEATURES):

    auc_drops = []
    ap_drops = []

    for repeat in range(N_REPEATS):

        rng = np.random.default_rng(
            SEED + feature_index * 1000 + repeat
        )

        X_permuted = X_test_scaled.copy()

        X_permuted[:, feature_index] = rng.permutation(
            X_permuted[:, feature_index]
        )

        perturbed_prob = predict_probabilities(
            X_permuted
        )

        perturbed_auc = roc_auc_score(
            y_test, perturbed_prob
        )

        perturbed_ap = average_precision_score(
            y_test, perturbed_prob
        )

        auc_drops.append(
            baseline_auc - perturbed_auc
        )

        ap_drops.append(
            baseline_ap - perturbed_ap
        )

    result = {
        "model": "MLP",
        "feature": feature_name,
        "n_repeats": N_REPEATS,
        "baseline_auroc": baseline_auc,
        "baseline_auprc": baseline_ap,
        "mean_auroc_drop": float(np.mean(auc_drops)),
        "sd_auroc_drop": float(
            np.std(auc_drops, ddof=1)
        ),
        "mean_auprc_drop": float(np.mean(ap_drops)),
        "sd_auprc_drop": float(
            np.std(ap_drops, ddof=1)
        )
    }

    results.append(result)

    print(
        feature_name,
        "| AUROC drop:",
        round(result["mean_auroc_drop"], 6),
        "| AUPRC drop:",
        round(result["mean_auprc_drop"], 6),
        flush=True
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

pd.DataFrame(results).to_csv(
    OUTPUT_DIR / "rq6_mlp_permutation_importance.csv",
    index=False
)

print()
print("RQ6 MLP COMPLETED", flush=True)
print("Outputs:", OUTPUT_DIR.resolve(), flush=True)

