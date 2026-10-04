import warnings
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

surv = pd.read_parquet(
    "cohort_survival_final_365d.parquet"
)

feat = pd.read_parquet(
    "structured_features_365d_safe.parquet"
)

FEATURES = [
    "prior_dx_count",
    "prior_unique_dx_count",
    "prior_visit_count",
    "age_at_last_prior_visit",
    "mean_codes_per_visit",
    "max_codes_per_visit",
    "mean_unique_codes_per_visit",
    "prior_icd9_count",
    "prior_icd10_count",
]

df = (
    surv[
        [
            "subject_id",
            "time_to_dn_days",
            "dn_event",
            "split"
        ]
    ]
    .merge(
        feat[
            ["subject_id"] + FEATURES
        ],
        on="subject_id",
        how="inner"
    )
)

print("Patients:", len(df))
print("Events:", int(df["dn_event"].sum()))
print("Censored:", int((df["dn_event"] == 0).sum()))


# ============================================================
# 2. FIXED SPLITS
# ============================================================

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

print("\nTrain:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print(
    "Train events:",
    int(train["dn_event"].sum())
)

print(
    "Validation events:",
    int(val["dn_event"].sum())
)

print(
    "Test events:",
    int(test["dn_event"].sum())
)


# ============================================================
# 3. TRAIN-ONLY PREPROCESSING
# ============================================================

# Median imputation learned from training data only.
train_medians = train[FEATURES].median()

for part in [train, val, test]:
    part[FEATURES] = (
        part[FEATURES]
        .fillna(train_medians)
    )

# Standardisation learned from training data only.
train_means = train[FEATURES].mean()
train_stds = train[FEATURES].std()

train_stds = train_stds.replace(
    0,
    1.0
)

for part in [train, val, test]:
    part[FEATURES] = (
        part[FEATURES] - train_means
    ) / train_stds


# ============================================================
# 4. PENALIZER SELECTION ON VALIDATION SET
# ============================================================

PENALIZERS = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    1.0
]

best_cindex = -np.inf
best_penalizer = None
best_model = None

train_fit = train[
    FEATURES
    + [
        "time_to_dn_days",
        "dn_event"
    ]
].copy()

for penalizer in PENALIZERS:

    try:

        model = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=0.0
        )

        model.fit(
            train_fit,
            duration_col="time_to_dn_days",
            event_col="dn_event",
            show_progress=False
        )

        val_hazard = (
            model.predict_partial_hazard(
                val[FEATURES]
            )
            .to_numpy()
            .reshape(-1)
        )

        # lifelines concordance_index expects larger
        # scores to correspond to longer survival.
        val_cindex = concordance_index(
            val["time_to_dn_days"],
            -val_hazard,
            val["dn_event"]
        )

        print(
            f"Penalizer {penalizer:<7} "
            f"| Val C-index {val_cindex:.6f}"
        )

        if val_cindex > best_cindex:

            best_cindex = val_cindex
            best_penalizer = penalizer
            best_model = model

    except Exception as e:

        print(
            "Penalizer",
            penalizer,
            "FAILED:",
            e
        )


if best_model is None:
    raise RuntimeError(
        "No Cox model fitted successfully."
    )


print(
    "\nSelected penalizer:",
    best_penalizer
)

print(
    "Best validation C-index:",
    round(best_cindex, 6)
)


# ============================================================
# 5. TEST PREDICTIONS
# ============================================================

test_hazard = (
    best_model.predict_partial_hazard(
        test[FEATURES]
    )
    .to_numpy()
    .reshape(-1)
)

test_cindex = concordance_index(
    test["time_to_dn_days"],
    -test_hazard,
    test["dn_event"]
)


# ============================================================
# 6. 365-DAY ABSOLUTE RISK
# ============================================================

survival_365 = (
    best_model.predict_survival_function(
        test[FEATURES],
        times=[365.0]
    )
)

test_risk_365 = (
    1.0
    - survival_365.iloc[0].to_numpy()
)

test_auc = roc_auc_score(
    test["dn_event"],
    test_risk_365
)

test_auprc = average_precision_score(
    test["dn_event"],
    test_risk_365
)


# ============================================================
# 7. RESULTS
# ============================================================

results = {
    "Model":
        "Cox PH Survival 365d",

    "N_test":
        len(test),

    "Events_test":
        int(test["dn_event"].sum()),

    "C_index":
        test_cindex,

    "AUROC_365d":
        test_auc,

    "AUPRC_365d":
        test_auprc,

    "Best_Val_C_index":
        best_cindex,

    "Selected_penalizer":
        best_penalizer,
}


print()
print("=" * 60)
print("COX PH TIME-TO-FIRST-DOCUMENTED-DN RESULTS")
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
# 8. SAVE RESULTS
# ============================================================

pd.DataFrame(
    [results]
).to_csv(
    "survival_coxph_365d_results.csv",
    index=False
)

pd.DataFrame(
    {
        "subject_id":
            test["subject_id"].to_numpy(),

        "time_to_dn_days":
            test["time_to_dn_days"].to_numpy(),

        "dn_event":
            test["dn_event"].to_numpy(),

        "predicted_365d_risk":
            test_risk_365,

        "partial_hazard":
            test_hazard,
    }
).to_csv(
    "survival_coxph_365d_predictions.csv",
    index=False
)


# ============================================================
# 9. SAVE COEFFICIENTS / HAZARD RATIOS
# ============================================================

coef = (
    best_model.summary[
        [
            "coef",
            "exp(coef)",
            "se(coef)",
            "p"
        ]
    ]
    .reset_index()
)

coef.to_csv(
    "survival_coxph_365d_coefficients.csv",
    index=False
)

print()
print(
    "Saved: survival_coxph_365d_results.csv"
)

print(
    "Saved: survival_coxph_365d_predictions.csv"
)

print(
    "Saved: survival_coxph_365d_coefficients.csv"
)
