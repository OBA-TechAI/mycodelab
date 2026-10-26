
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
N_REPEATS = 10

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("rq6_structured_ml_outputs")
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
# 2. LOAD LEAKAGE-SAFE DATA
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
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = train_df[FEATURES].astype(float).values
X_test = test_df[FEATURES].astype(float).values

y_train = train_df[TARGET].astype(int).values
y_test = test_df[TARGET].astype(int).values

test_ids = test_df["subject_id"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

positive = int(y_train.sum())
negative = int((y_train == 0).sum())
class_ratio = negative / positive

print("Total patients:", len(df), flush=True)
print("Train:", len(train_df), flush=True)
print("Validation:", len(val_df), flush=True)
print("Test:", len(test_df), flush=True)
print("Test DN events:", int(y_test.sum()), flush=True)


# ============================================================
# 3. RECONSTRUCT ORIGINAL MODELS
# ============================================================

models = {
    "Logistic Regression": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED
        ),
        X_train_scaled,
        X_test_scaled
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),
        X_train,
        X_test
    ),

    "XGBoost": (
        XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=class_ratio,
            random_state=SEED,
            n_jobs=4
        ),
        X_train,
        X_test
    )
}


# ============================================================
# 4. PERMUTATION FEATURE IMPORTANCE
# ============================================================

importance_results = []
baseline_results = []

prediction_df = pd.DataFrame({
    "subject_id": test_ids,
    "label": y_test
})

for model_index, (
    model_name,
    (model, X_fit, X_eval)
) in enumerate(models.items()):

    print("\n" + "=" * 60, flush=True)
    print("Training:", model_name, flush=True)

    model.fit(X_fit, y_train)

    baseline_prob = model.predict_proba(X_eval)[:, 1]

    baseline_auc = roc_auc_score(y_test, baseline_prob)
    baseline_ap = average_precision_score(
        y_test, baseline_prob
    )

    print("Baseline AUROC:", baseline_auc, flush=True)
    print("Baseline AUPRC:", baseline_ap, flush=True)

    baseline_results.append({
        "model": model_name,
        "n_test": len(y_test),
        "dn_events": int(y_test.sum()),
        "auroc": baseline_auc,
        "auprc": baseline_ap
    })

    probability_column = {
        "Logistic Regression": "logistic_probability",
        "Random Forest": "random_forest_probability",
        "XGBoost": "xgboost_probability"
    }[model_name]

    prediction_df[probability_column] = baseline_prob

    for feature_index, feature_name in enumerate(FEATURES):

        auc_drops = []
        ap_drops = []

        for repeat in range(N_REPEATS):

            # Deterministic, independent shuffle for each
            # model, feature and repeat.
            rng = np.random.default_rng(
                SEED
                + model_index * 100000
                + feature_index * 1000
                + repeat
            )

            X_permuted = X_eval.copy()

            X_permuted[:, feature_index] = (
                rng.permutation(
                    X_permuted[:, feature_index]
                )
            )

            permuted_prob = model.predict_proba(
                X_permuted
            )[:, 1]

            permuted_auc = roc_auc_score(
                y_test, permuted_prob
            )

            permuted_ap = average_precision_score(
                y_test, permuted_prob
            )

            auc_drops.append(
                baseline_auc - permuted_auc
            )

            ap_drops.append(
                baseline_ap - permuted_ap
            )

        importance_results.append({
            "model": model_name,
            "feature": feature_name,
            "n_repeats": N_REPEATS,
            "baseline_auroc": baseline_auc,
            "baseline_auprc": baseline_ap,
            "mean_auroc_drop": float(
                np.mean(auc_drops)
            ),
            "sd_auroc_drop": float(
                np.std(auc_drops, ddof=1)
            ),
            "mean_auprc_drop": float(
                np.mean(ap_drops)
            ),
            "sd_auprc_drop": float(
                np.std(ap_drops, ddof=1)
            )
        })

        print(
            feature_name,
            "| AUROC drop:",
            round(np.mean(auc_drops), 6),
            "| AUPRC drop:",
            round(np.mean(ap_drops), 6),
            flush=True
        )


# ============================================================
# 5. SAVE RESULTS
# ============================================================

pd.DataFrame(baseline_results).to_csv(
    OUTPUT_DIR / "rq6_structured_ml_baselines.csv",
    index=False
)

importance_df = pd.DataFrame(importance_results)

importance_df.to_csv(
    OUTPUT_DIR / "rq6_structured_ml_permutation_importance.csv",
    index=False
)

prediction_df.to_csv(
    OUTPUT_DIR / "rq6_structured_ml_reconstructed_predictions.csv",
    index=False
)

print("\nRQ6 STRUCTURED ML COMPLETED", flush=True)
print("Outputs:", OUTPUT_DIR.resolve(), flush=True)

