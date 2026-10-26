import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.multitest import multipletests

INPUT = "final_5seed_all_model_predictions_long_corrected.csv"
B = 30000
RANDOM_SEED = 42

df = pd.read_csv(INPUT)

models = sorted(df["model"].unique())
seeds = sorted(df["seed"].unique())
patients = np.sort(df["subject_id"].unique())

assert len(models) == 26
assert len(seeds) == 5
assert len(patients) == 1337
assert not df.duplicated(["model", "seed", "subject_id"]).any()

reference = (
    df[(df.model == models[0]) & (df.seed == seeds[0])]
    .set_index("subject_id")
    .loc[patients]
)

y = reference["label"].to_numpy(dtype=int)

pred = {}

for model in models:
    pred[model] = np.stack([
        df[(df.model == model) & (df.seed == seed)]
        .set_index("subject_id")
        .loc[patients]["probability"].to_numpy(dtype=float)
        for seed in seeds
    ])

positive = np.flatnonzero(y == 1)
negative = np.flatnonzero(y == 0)

rng = np.random.default_rng(RANDOM_SEED)

# Precompute shared resampling indices
pos_samples = rng.choice(
    positive, size=(B, len(positive)), replace=True
)
neg_samples = rng.choice(
    negative, size=(B, len(negative)), replace=True
)

samples = np.concatenate(
    [pos_samples, neg_samples], axis=1
)

del pos_samples, neg_samples


# Efficient batched bootstrap scoring.
# Shared patient samples preserve pairing across models.

scores = {}
batch_size = 100

for k, model in enumerate(models, 1):
    p = pred[model]

    observed_auc = np.mean([
        roc_auc_score(y, row) for row in p
    ])
    observed_ap = np.mean([
        average_precision_score(y, row) for row in p
    ])

    boot_auc = np.zeros(B)
    boot_ap = np.zeros(B)

    for start_idx in range(0, B, batch_size):
        end_idx = min(start_idx + batch_size, B)
        idx = samples[start_idx:end_idx]

        yy = y[idx]
        n_pos = yy.sum(axis=1)
        n_neg = yy.shape[1] - n_pos

        for seed_predictions in p:
            pp = seed_predictions[idx]

            # Stable descending order for tied scores
            order = np.argsort(-pp, axis=1, kind="stable")
            sorted_y = np.take_along_axis(yy, order, axis=1)
            sorted_p = np.take_along_axis(pp, order, axis=1)

            tp = np.cumsum(sorted_y, axis=1)
            fp = np.cumsum(1 - sorted_y, axis=1)

            # AUROC using grouped thresholds to handle ties.
            # Count positive-negative pairs with half-credit for ties.
            last_in_group = np.ones_like(sorted_y, dtype=bool)
            last_in_group[:, :-1] = (
                sorted_p[:, :-1] != sorted_p[:, 1:]
            )

            group_starts_auc = np.concatenate(
                [
                    np.ones((len(idx), 1), dtype=bool),
                    sorted_p[:, 1:] != sorted_p[:, :-1]
                ],
                axis=1
            )

            start_positions = np.maximum.accumulate(
                np.where(
                    group_starts_auc,
                    np.arange(sorted_p.shape[1])[None, :],
                    0
                ),
                axis=1
            )

            tp_before = np.take_along_axis(
                np.concatenate(
                    [np.zeros((len(idx), 1)), tp[:, :-1]],
                    axis=1
                ),
                start_positions,
                axis=1
            )

            fp_before = np.take_along_axis(
                np.concatenate(
                    [np.zeros((len(idx), 1)), fp[:, :-1]],
                    axis=1
                ),
                start_positions,
                axis=1
            )

            group_tp = tp - tp_before
            group_fp = fp - fp_before

            concordant = (
                group_tp
                * (n_neg[:, None] - fp)
            )

            tied = 0.5 * group_tp * group_fp

            auc = (
                ((concordant + tied) * last_in_group).sum(axis=1)
                / (n_pos * n_neg)
            )

            # Average precision at distinct score thresholds
            precision = tp / (tp + fp)

            last_in_tie = np.ones_like(sorted_y, dtype=bool)
            last_in_tie[:, :-1] = (
                sorted_p[:, :-1] != sorted_p[:, 1:]
            )

            # For tied groups, AP must use the total positive
            # mass within each group, not individual positions.
            group_starts = np.concatenate(
                [
                    np.ones((len(idx), 1), dtype=bool),
                    sorted_p[:, 1:] != sorted_p[:, :-1]
                ],
                axis=1
            )

            start_indices = np.maximum.accumulate(
                np.where(
                    group_starts,
                    np.arange(sorted_p.shape[1])[None, :],
                    0
                ),
                axis=1
            )

            before_group = np.take_along_axis(
                np.concatenate(
                    [
                        np.zeros((len(idx), 1)),
                        tp[:, :-1]
                    ],
                    axis=1
                ),
                start_indices,
                axis=1
            )

            group_positive = tp - before_group

            ap = (
                precision
                * group_positive
                * last_in_tie
            ).sum(axis=1) / n_pos

            boot_auc[start_idx:end_idx] += auc / len(seeds)
            boot_ap[start_idx:end_idx] += ap / len(seeds)

    scores[model] = {
        "AUROC": np.concatenate([[observed_auc], boot_auc]),
        "AUPRC": np.concatenate([[observed_ap], boot_ap])
    }

    print(f"Completed {k}/26: {model}", flush=True)

rows = []

for metric in ["AUROC", "AUPRC"]:
    for a, b in combinations(models, 2):
        difference = scores[a][metric] - scores[b][metric]
        observed = difference[0]
        boot = difference[1:]

        lower, upper = np.percentile(boot, [2.5, 97.5])

        # Null-centred bootstrap distribution
        centred = boot - boot.mean()
        p_value = (
            np.count_nonzero(np.abs(centred) >= abs(observed)) + 1
        ) / (B + 1)

        rows.append({
            "Metric": metric,
            "Model_A": a,
            "Model_B": b,
            "Difference_A_minus_B": observed,
            "CI_Lower": lower,
            "CI_Upper": upper,
            "P_Unadjusted": p_value
        })

result = pd.DataFrame(rows)
result["P_Holm"] = np.nan
result["Significant_Holm_0.05"] = False

for metric in ["AUROC", "AUPRC"]:
    mask = result["Metric"] == metric

    reject, adjusted, _, _ = multipletests(
        result.loc[mask, "P_Unadjusted"],
        alpha=0.05,
        method="holm"
    )

    result.loc[mask, "P_Holm"] = adjusted
    result.loc[mask, "Significant_Holm_0.05"] = reject

result.to_csv(
    "rq7_all325_paired_bootstrap_30000_holm.csv",
    index=False
)

result[result["Significant_Holm_0.05"]].to_csv(
    "rq7_all325_significant_30000.csv",
    index=False
)

print("\n===== FINAL RQ7 STATISTICAL SUMMARY =====")
print("Models:", len(models))
print("Model pairs:", 325)
print("Total tests:", len(result))
print("Bootstrap replicates:", B)

for metric in ["AUROC", "AUPRC"]:
    subset = result[result.Metric == metric]
    print(
        metric,
        "Holm-significant:",
        int(subset["Significant_Holm_0.05"].sum()),
        "Minimum adjusted p:",
        subset.P_Holm.min()
    )

print("\nSaved: rq7_all325_paired_bootstrap_30000_holm.csv")
print("Saved: rq7_all325_significant_30000.csv")
