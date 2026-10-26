import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
REPEATS = 30
HIDDEN, N_HEADS, N_LAYERS, DROPOUT = 128, 8, 4, 0.1
FEATURES = [
    'prior_dx_count', 'prior_unique_dx_count', 'prior_visit_count',
    'age_at_last_prior_visit', 'mean_codes_per_visit', 'max_codes_per_visit',
    'mean_unique_codes_per_visit', 'prior_icd9_count', 'prior_icd10_count',
    'mean_intervisit_gap_days', 'max_intervisit_gap_days',
    'last_intervisit_gap_days', 'last_visit_recency_days', 'history_span_days'
]
TEMPORAL = FEATURES[9:]
CHECKPOINT = 'timeaware_fttransformer_365d_best.pt'
OUT = Path('rq6_timeaware_fttransformer_outputs')
OUT.mkdir(exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cpu')  # CPU inference works on login nodes; no GPU allocation needed

class FTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_features = len(FEATURES)
        self.feature_weight = nn.Parameter(torch.randn(self.n_features, HIDDEN) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(self.n_features, HIDDEN))
        self.feature_embedding = nn.Embedding(self.n_features, HIDDEN)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=N_HEADS, dim_feedforward=HIDDEN * 4,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        feature_ids = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.feature_embedding(feature_ids).unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        output = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.classifier(self.norm(output[:, 0])).squeeze(-1)

print('Loading data...', flush=True)
df = pd.read_parquet('structured_features_timeaware_365d_safe.parquet')
splits = pd.read_parquet('patient_splits_timeaware_365d.parquet')
df = df.merge(splits, on='subject_id', how='inner', validate='one_to_one')
train = df.loc[df['split'] == 'train'].copy()
test = df.loc[df['split'] == 'test'].copy()
if len(train) == 0 or len(test) != 1337:
    raise RuntimeError(f'Unexpected split sizes: train={len(train)}, test={len(test)}')
scaler = StandardScaler()
scaler.fit(train[FEATURES])
X = scaler.transform(test[FEATURES]).astype('float32')
y = test['dn_within_365d'].to_numpy().astype(int)
if int(y.sum()) != 103:
    raise RuntimeError(f'Expected 103 test DN events; found {int(y.sum())}')

model = FTTransformer().to(DEVICE)
try:
    weights = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
except TypeError:
    weights = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(weights, strict=True)
model.eval()

def predict(x):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            tensor = torch.from_numpy(x[start:start + 64]).to(DEVICE)
            chunks.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(chunks)

def metrics(prob):
    return roc_auc_score(y, prob), average_precision_score(y, prob)

baseline_probs = predict(X)
base_auc, base_ap = metrics(baseline_probs)
print(f'Reproduced baseline: AUROC={base_auc:.6f}, AUPRC={base_ap:.6f}', flush=True)
original = pd.read_csv('timeaware_fttransformer_365d_results.csv').iloc[0]
print('Original result columns:', list(original.index), flush=True)

# ============================================================
# RQ6: TIME-AWARE FT-TRANSFORMER FEATURE IMPORTANCE
# ============================================================

from pathlib import Path

OUTPUT = Path("rq6_timeaware_fttransformer_outputs")
OUTPUT.mkdir(exist_ok=True)

REPEATS = 30
rng = np.random.default_rng(2026)

# Verify reproduced baseline against original results
expected_auc = float(original["AUROC"])
expected_ap = float(original["AUPRC"])

if abs(base_auc - expected_auc) > 0.001 or abs(base_ap - expected_ap) > 0.001:
    raise RuntimeError("Baseline mismatch. Check model and preprocessing.")

TEMPORAL_FEATURES = [
    "mean_intervisit_gap_days",
    "max_intervisit_gap_days",
    "last_intervisit_gap_days",
    "last_visit_recency_days",
    "history_span_days"
]

records = []

for j, feature in enumerate(FEATURES):
    for repeat in range(1, REPEATS + 1):
        X_perm = X.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])

        perm_auc, perm_ap = metrics(predict(X_perm))

        records.append({
            "feature": feature,
            "group": "Temporal" if feature in TEMPORAL_FEATURES else "Clinical history",
            "repeat": repeat,
            "AUROC_drop": base_auc - perm_auc,
            "AUPRC_drop": base_ap - perm_ap
        })

    print("Finished:", feature, flush=True)

repeats_df = pd.DataFrame(records)
repeats_df.to_csv(
    OUTPUT / "rq6_timeaware_fttransformer_permutation_repeats.csv",
    index=False
)

summary = repeats_df.groupby(["feature", "group"]).agg(
    AUROC_mean=("AUROC_drop", "mean"),
    AUROC_std=("AUROC_drop", "std"),
    AUPRC_mean=("AUPRC_drop", "mean"),
    AUPRC_std=("AUPRC_drop", "std")
).reset_index()

summary = summary.sort_values("AUROC_mean", ascending=False)
summary.to_csv(
    OUTPUT / "rq6_timeaware_fttransformer_feature_importance.csv",
    index=False
)

pd.DataFrame([{
    "AUROC": base_auc,
    "AUPRC": base_ap,
    "N_test": len(y)
}]).to_csv(
    OUTPUT / "rq6_timeaware_fttransformer_baseline.csv",
    index=False
)

# Ranked bar charts
for metric in ["AUROC", "AUPRC"]:
    ordered = summary.sort_values(f"{metric}_mean")
    plt.figure(figsize=(12, 7))
    plt.barh(
        ordered["feature"],
        ordered[f"{metric}_mean"],
        xerr=ordered[f"{metric}_std"],
        capsize=3
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"Mean reduction in {metric}")
    plt.title(f"Time-Aware FT-Transformer: {metric} Permutation Importance")
    plt.tight_layout()
    plt.savefig(
        OUTPUT / f"rq6_timeaware_fttransformer_{metric.lower()}_bar.png",
        dpi=200
    )
    plt.close()

# Line plot comparing AUROC and AUPRC importance
plt.figure(figsize=(13, 6))
positions = np.arange(len(summary))
plt.plot(positions, summary["AUROC_mean"], marker="o", label="AUROC")
plt.plot(positions, summary["AUPRC_mean"], marker="s", label="AUPRC")
plt.xticks(positions, summary["feature"], rotation=45, ha="right")
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Mean performance reduction")
plt.title("Time-Aware FT-Transformer: Feature Importance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT / "rq6_timeaware_fttransformer_comparison_line.png",
    dpi=200
)
plt.close()

# Clinical-history versus temporal-feature comparison
group_summary = summary.groupby("group")[
    ["AUROC_mean", "AUPRC_mean"]
].mean()

group_summary.to_csv(
    OUTPUT / "rq6_timeaware_fttransformer_group_comparison.csv"
)

group_summary.plot(kind="bar", figsize=(9, 6))
plt.ylabel("Mean individual-feature importance")
plt.title("Clinical History vs Temporal Features")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUTPUT / "rq6_timeaware_fttransformer_group_comparison.png",
    dpi=200
)
plt.close()

print("\nRQ6 Time-Aware FT-Transformer analysis completed.")
print(summary.to_string(index=False))
print("\nGroup comparison:")
print(group_summary.to_string())
print("\nSaved results to:", OUTPUT.resolve())

