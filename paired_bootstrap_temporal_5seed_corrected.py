import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

FILE = "final_5seed_all_model_predictions_long_corrected.csv"

SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 2000
RANDOM_SEED = 2026

PAIRS = [
    ("BEHRT",
     "BEHRT Control",
     "BEHRT Time-Aware"),

    ("Med-BERT",
     "Med-BERT Control",
     "Med-BERT Time-Aware"),

    ("CEHR-BERT",
     "CEHR-BERT Control",
     "CEHR-BERT Time-Aware"),

    ("FT-Transformer",
     "FT-Transformer Control",
     "FT-Transformer Time-Aware"),

    ("TransTab",
     "TransTab Control",
     "TransTab Time-Aware"),

    ("BioBERT",
     "BioBERT Control",
     "BioBERT Time-Aware"),

    ("ClinicalBERT",
     "ClinicalBERT Control",
     "ClinicalBERT Time-Aware"),

    ("Clinical Longformer",
     "Clinical Longformer Control",
     "Clinical Longformer Time-Aware"),

    ("LSTM",
     "LSTM Control",
     "LSTM Time-Aware"),

    ("GRU",
     "GRU Control",
     "GRU Time-Aware"),

    ("Llama 3.2 3B LoRA",
     "Llama 3.2 3B LoRA Control",
     "Llama 3.2 3B LoRA Time-Aware"),
]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(m)

    previous = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        value = max(value, previous)
        value = min(value, 1.0)

        adjusted[idx] = value
        previous = value

    return adjusted


df = pd.read_csv(FILE)

print("Rows:", len(df))
print("Models:", df["model"].nunique())
print("Seeds:", sorted(df["seed"].unique()))

if not np.isfinite(df["probability"]).all():
    raise RuntimeError("Non-finite probabilities found.")

# ------------------------------------------------------------
# Establish common patient order and labels
# ------------------------------------------------------------
reference = (
    df[
        (df["model"] == "BEHRT Control")
        & (df["seed"] == 42)
    ]
    .sort_values("subject_id")
)

subject_ids = reference["subject_id"].to_numpy()
y = reference["label"].to_numpy().astype(int)

print("Test patients:", len(y))
print("Positive:", int(y.sum()))
print("Negative:", int((1-y).sum()))

pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# ------------------------------------------------------------
# Build probability matrices:
# model -> [5 seeds, 1337 patients]
# ------------------------------------------------------------
probabilities = {}

for model in sorted(df["model"].unique()):

    seed_arrays = []

    for seed in SEEDS:

        g = (
            df[
                (df["model"] == model)
                & (df["seed"] == seed)
            ]
            .sort_values("subject_id")
        )

        if len(g) != len(y):
            raise RuntimeError(
                f"{model}, seed {seed}: "
                f"expected {len(y)} patients, got {len(g)}"
            )

        if not np.array_equal(
            g["subject_id"].to_numpy(),
            subject_ids
        ):
            raise RuntimeError(
                f"Patient mismatch: {model}, seed {seed}"
            )

        if not np.array_equal(
            g["label"].to_numpy().astype(int),
            y
        ):
            raise RuntimeError(
                f"Label mismatch: {model}, seed {seed}"
            )

        seed_arrays.append(
            g["probability"].to_numpy(dtype=float)
        )

    probabilities[model] = np.vstack(seed_arrays)


def mean_metric(prob_matrix, indices, metric):

    values = []

    yy = y[indices]

    for row in prob_matrix:

        pp = row[indices]

        if metric == "AUROC":
            value = roc_auc_score(yy, pp)

        elif metric == "AUPRC":
            value = average_precision_score(
                yy,
                pp
            )

        else:
            raise ValueError(metric)

        values.append(value)

    return float(np.mean(values))


rng = np.random.default_rng(RANDOM_SEED)

results = []

for family, control, temporal in PAIRS:

    print("\n========================================")
    print(family)
    print("========================================")

    ctrl = probabilities[control]
    temp = probabilities[temporal]

    full_idx = np.arange(len(y))

    for metric in ["AUROC", "AUPRC"]:

        control_observed = mean_metric(
            ctrl,
            full_idx,
            metric
        )

        temporal_observed = mean_metric(
            temp,
            full_idx,
            metric
        )

        observed_delta = (
            temporal_observed
            - control_observed
        )

        boot_deltas = np.empty(
            N_BOOT,
            dtype=float
        )

        for b in range(N_BOOT):

            sampled_pos = rng.choice(
                pos_idx,
                size=len(pos_idx),
                replace=True
            )

            sampled_neg = rng.choice(
                neg_idx,
                size=len(neg_idx),
                replace=True
            )

            indices = np.concatenate(
                [sampled_pos, sampled_neg]
            )

            ctrl_metric = mean_metric(
                ctrl,
                indices,
                metric
            )

            temp_metric = mean_metric(
                temp,
                indices,
                metric
            )

            boot_deltas[b] = (
                temp_metric
                - ctrl_metric
            )

        ci_low, ci_high = np.percentile(
            boot_deltas,
            [2.5, 97.5]
        )

        # Approximate two-sided paired-bootstrap p-value
        p_lower = (
            np.sum(boot_deltas <= 0) + 1
        ) / (N_BOOT + 1)

        p_upper = (
            np.sum(boot_deltas >= 0) + 1
        ) / (N_BOOT + 1)

        p_value = min(
            1.0,
            2 * min(p_lower, p_upper)
        )

        print(
            f"{metric}: "
            f"Control={control_observed:.6f} | "
            f"Time-aware={temporal_observed:.6f} | "
            f"Delta={observed_delta:+.6f} | "
            f"95% CI [{ci_low:+.6f}, {ci_high:+.6f}] | "
            f"p={p_value:.6f}"
        )

        results.append({
            "Family": family,
            "Metric": metric,

            "Control_Model": control,
            "TimeAware_Model": temporal,

            "Control_Mean_5Seed":
                control_observed,

            "TimeAware_Mean_5Seed":
                temporal_observed,

            "Delta_TimeAware_minus_Control":
                observed_delta,

            "Bootstrap_95CI_Low":
                ci_low,

            "Bootstrap_95CI_High":
                ci_high,

            "Bootstrap_p":
                p_value,

            "N_Bootstrap":
                N_BOOT,

            "N_Test":
                len(y),

            "N_Positive":
                int(y.sum()),

            "N_Negative":
                int((1-y).sum())
        })


results = pd.DataFrame(results)

# ------------------------------------------------------------
# Holm correction separately for AUROC and AUPRC
# ------------------------------------------------------------
results["Holm_Adjusted_p"] = np.nan

for metric in ["AUROC", "AUPRC"]:

    mask = results["Metric"] == metric

    results.loc[
        mask,
        "Holm_Adjusted_p"
    ] = holm_adjust(
        results.loc[
            mask,
            "Bootstrap_p"
        ].values
    )


results["Significant_0.05"] = (
    results["Holm_Adjusted_p"] < 0.05
)

results.to_csv(
    "paired_bootstrap_temporal_5seed_corrected_results.csv",
    index=False
)


print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(
    results[
        [
            "Family",
            "Metric",
            "Control_Mean_5Seed",
            "TimeAware_Mean_5Seed",
            "Delta_TimeAware_minus_Control",
            "Bootstrap_95CI_Low",
            "Bootstrap_95CI_High",
            "Bootstrap_p",
            "Holm_Adjusted_p",
            "Significant_0.05"
        ]
    ].to_string(index=False)
)

print(
    "\nSaved:",
    "paired_bootstrap_temporal_5seed_corrected_results.csv"
)
