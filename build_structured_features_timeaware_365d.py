import pandas as pd
import numpy as np

BASE_FILE = "structured_features_365d_safe.parquet"
SEQ_FILE = "dx_sequences_timeaware_behrt_365d.parquet"
OUT_FILE = "structured_features_timeaware_365d_safe.parquet"

base = pd.read_parquet(BASE_FILE)
seq = pd.read_parquet(SEQ_FILE)

# ------------------------------------------------------------
# One row per historical admission.
# Important: temporal values are repeated for diagnosis codes,
# so deduplicate visits before calculating temporal summaries.
# ------------------------------------------------------------
visits = (
    seq[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "prediction_cutoff",
            "visit_number",
            "time_gap_days",
            "time_to_cutoff_days",
        ]
    ]
    .drop_duplicates(subset=["subject_id", "hadm_id"])
    .sort_values(["subject_id", "admittime"])
)

def temporal_features(g):

    g = g.sort_values("admittime")

    gaps = g["time_gap_days"].astype(float)

    # Exclude first-visit zero/undefined interval from inter-visit summaries
    positive_gaps = gaps[gaps > 0]

    if len(positive_gaps) > 0:
        mean_gap = positive_gaps.mean()
        max_gap = positive_gaps.max()
        last_gap = positive_gaps.iloc[-1]
    else:
        mean_gap = 0.0
        max_gap = 0.0
        last_gap = 0.0

    # Closest historical visit to prediction cutoff
    last_visit_recency = g["time_to_cutoff_days"].astype(float).min()

    # Span of available historical record
    if len(g) > 1:
        history_span = (
            pd.to_datetime(g["admittime"]).max()
            - pd.to_datetime(g["admittime"]).min()
        ).days
    else:
        history_span = 0.0

    return pd.Series(
        {
            "mean_intervisit_gap_days": mean_gap,
            "max_intervisit_gap_days": max_gap,
            "last_intervisit_gap_days": last_gap,
            "last_visit_recency_days": last_visit_recency,
            "history_span_days": float(history_span),
        }
    )

temporal = (
    visits.groupby("subject_id", group_keys=False)
    .apply(temporal_features)
    .reset_index()
)

final = base.merge(
    temporal,
    on="subject_id",
    how="left",
    validate="one_to_one"
)

temporal_cols = [
    "mean_intervisit_gap_days",
    "max_intervisit_gap_days",
    "last_intervisit_gap_days",
    "last_visit_recency_days",
    "history_span_days",
]

final[temporal_cols] = final[temporal_cols].fillna(0.0)

assert len(final) == len(base)
assert final["subject_id"].nunique() == 8907
assert final["dn_within_365d"].sum() == 688

final.to_parquet(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Shape:", final.shape)
print("Patients:", final["subject_id"].nunique())
print("DN positive:", int(final["dn_within_365d"].sum()))
print("\nColumns:")
for c in final.columns:
    print(c)
