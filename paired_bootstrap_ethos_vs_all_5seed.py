import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# CONFIG
# ============================================================

INPUT = (
    "final_5seed_all_model_predictions_long_with_ethos.csv"
)

OUTPUT = (
    "paired_bootstrap_ethos_vs_all_5seed_results.csv"
)

ANCHOR = "ETHOS Pretrained"

SEEDS = [42, 43, 44, 45, 46]

N_BOOT = 2000
RANDOM_SEED = 2026
N_JOBS = 8


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

models = sorted(df["model"].unique())

assert ANCHOR in models
assert len(models) == 27

comparators = [
    m for m in models
    if m != ANCHOR
]

print("Models:", len(models))
print("Comparators against ETHOS:", len(comparators))


# ============================================================
# ALIGN TEST PATIENTS
# ============================================================

reference = (
    df[
        (df["model"] == ANCHOR)
        & (df["seed"] == SEEDS[0])
    ]
    .sort_values("subject_id")
    .reset_index(drop=True)
)

subject_ids = reference["subject_id"].to_numpy()
y = reference["label"].to_numpy(dtype=int)

n = len(subject_ids)

print("Test patients:", n)
print("Positives:", int(y.sum()))
print("Negatives:", int((1-y).sum()))

assert n == 1337


# ============================================================
# BUILD PROBABILITY MATRICES
# model -> [5 seeds, 1337 patients]
# ============================================================

probabilities = {}

for model in models:

    seed_probs = []

    for seed in SEEDS:

        part = (
            df[
                (df["model"] == model)
                & (df["seed"] == seed)
            ]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )

        assert len(part) == n
        assert np.array_equal(
            part["subject_id"].to_numpy(),
            subject_ids
        )

        assert np.array_equal(
            part["label"].to_numpy(dtype=int),
            y
        )

        seed_probs.append(
            part["probability"].to_numpy(dtype=float)
        )

    probabilities[model] = np.vstack(seed_probs)


# ============================================================
# STRATIFIED BOOTSTRAP INDICES
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

bootstrap_indices = []

for _ in range(N_BOOT):

    pos_sample = rng.choice(
        pos_idx,
        size=len(pos_idx),
        replace=True
    )

    neg_sample = rng.choice(
        neg_idx,
        size=len(neg_idx),
        replace=True
    )

    bootstrap_indices.append(
        np.concatenate(
            [pos_sample, neg_sample]
        )
    )


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def mean_seed_metric(
    probs,
    indices,
    metric
):

    yy = y[indices]

    values = []

    for s in range(len(SEEDS)):

        pp = probs[s, indices]

        if metric == "AUROC":
            value = roc_auc_score(
                yy,
                pp
            )
        else:
            value = average_precision_score(
                yy,
                pp
            )

        values.append(value)

    return float(
        np.mean(values)
    )


def observed_metric(
    probs,
    metric
):

    indices = np.arange(n)

    return mean_seed_metric(
        probs,
        indices,
        metric
    )


# ============================================================
# COMPUTE BOOTSTRAP DISTRIBUTIONS
# ============================================================

def calculate_model(model):

    print("Processing:", model)

    probs = probabilities[model]

    observed = {}

    boot = {}

    for metric in [
        "AUROC",
        "AUPRC",
    ]:

        observed[metric] = (
            observed_metric(
                probs,
                metric
            )
        )

        boot[metric] = np.asarray(
            [
                mean_seed_metric(
                    probs,
                    idx,
                    metric
                )
                for idx
                in bootstrap_indices
            ]
        )

    return (
        model,
        observed,
        boot
    )


calculated = Parallel(
    n_jobs=N_JOBS
)(
    delayed(calculate_model)(model)
    for model in models
)

observed = {}
boot_metrics = {}

for model, obs, boot in calculated:

    observed[model] = obs
    boot_metrics[model] = boot


# ============================================================
# BOOTSTRAP P VALUE
# ============================================================

def bootstrap_pvalue(delta):

    p_lower = (
        np.sum(delta <= 0) + 1
    ) / (len(delta) + 1)

    p_upper = (
        np.sum(delta >= 0) + 1
    ) / (len(delta) + 1)

    return min(
        1.0,
        2.0 * min(
            p_lower,
            p_upper
        )
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_adjust(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    m = len(pvalues)

    order = np.argsort(pvalues)

    adjusted = np.empty(
        m,
        dtype=float
    )

    previous = 0.0

    for rank, idx in enumerate(order):

        value = (
            (m - rank)
            * pvalues[idx]
        )

        value = min(
            1.0,
            max(
                previous,
                value
            )
        )

        adjusted[idx] = value

        previous = value

    return adjusted


# ============================================================
# ETHOS VS EACH COMPARATOR
# ============================================================

rows = []

for metric in [
    "AUROC",
    "AUPRC",
]:

    for comparator in comparators:

        delta = (
            boot_metrics[ANCHOR][metric]
            -
            boot_metrics[comparator][metric]
        )

        ci_low, ci_high = np.percentile(
            delta,
            [2.5, 97.5]
        )

        rows.append(
            {
                "metric": metric,
                "anchor": ANCHOR,
                "comparator": comparator,

                "anchor_mean":
                    observed[ANCHOR][metric],

                "comparator_mean":
                    observed[comparator][metric],

                "delta_ethos_minus_comparator":
                    observed[ANCHOR][metric]
                    -
                    observed[comparator][metric],

                "ci_low":
                    float(ci_low),

                "ci_high":
                    float(ci_high),

                "p_raw":
                    bootstrap_pvalue(
                        delta
                    ),
            }
        )


results = pd.DataFrame(rows)


# ============================================================
# HOLM SEPARATELY BY METRIC
# ============================================================

results["p_holm"] = np.nan

for metric in [
    "AUROC",
    "AUPRC",
]:

    mask = (
        results["metric"]
        == metric
    )

    results.loc[
        mask,
        "p_holm"
    ] = holm_adjust(
        results.loc[
            mask,
            "p_raw"
        ].to_numpy()
    )


results[
    "holm_significant_0_05"
] = (
    results["p_holm"]
    < 0.05
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("ETHOS VS ALL MODELS — PAIRED BOOTSTRAP COMPLETE")
print("=" * 80)

for metric in [
    "AUROC",
    "AUPRC",
]:

    print()
    print(
        "=====",
        metric,
        "====="
    )

    display = (
        results[
            results["metric"]
            == metric
        ]
        .sort_values(
            "delta_ethos_minus_comparator",
            ascending=False
        )
    )

    print(
        display[
            [
                "comparator",
                "anchor_mean",
                "comparator_mean",
                "delta_ethos_minus_comparator",
                "ci_low",
                "ci_high",
                "p_raw",
                "p_holm",
                "holm_significant_0_05",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("SAVED:", OUTPUT)
