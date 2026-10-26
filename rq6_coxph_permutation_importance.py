import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES = [
    'prior_dx_count', 'prior_unique_dx_count', 'prior_visit_count',
    'age_at_last_prior_visit', 'mean_codes_per_visit', 'max_codes_per_visit',
    'mean_unique_codes_per_visit', 'prior_icd9_count', 'prior_icd10_count'
]
PENALIZER = 0.0001
REPEATS = 30
SEED = 2026
OUT = Path('rq6_coxph_outputs')
OUT.mkdir(exist_ok=True)

surv = pd.read_parquet('cohort_survival_final_365d.parquet')
feat = pd.read_parquet('structured_features_365d_safe.parquet')
df = surv[['subject_id','time_to_dn_days','dn_event','split']].merge(
    feat[['subject_id'] + FEATURES], on='subject_id', how='inner')
train = df[df['split'] == 'train'].copy()
val = df[df['split'] == 'validation'].copy()
test = df[df['split'] == 'test'].copy()
medians = train[FEATURES].median()
for part in (train,val,test):
    part[FEATURES] = part[FEATURES].fillna(medians)
means = train[FEATURES].mean()
stds = train[FEATURES].std().replace(0, 1.0)
for part in (train,val,test):
    part[FEATURES] = (part[FEATURES] - means) / stds

model = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
model.fit(train[FEATURES + ['time_to_dn_days','dn_event']].copy(),
          duration_col='time_to_dn_days', event_col='dn_event', show_progress=False)

def metrics(x):
    hazard = model.predict_partial_hazard(x[FEATURES]).to_numpy().reshape(-1)
    risk = 1.0 - model.predict_survival_function(x[FEATURES], times=[365.0]).iloc[0].to_numpy()
    y = test['dn_event'].to_numpy()
    return {
        'C_index': concordance_index(test['time_to_dn_days'].to_numpy(), -hazard, y),
        'AUROC': roc_auc_score(y, risk),
        'AUPRC': average_precision_score(y, risk),
    }

base = metrics(test)
expected = {'C_index': 0.6980771410222507,
            'AUROC': 0.7043280200154207,
            'AUPRC': 0.1745358719519541}
print('Reproduced baseline:', base)
for k, v in expected.items():
    if not np.isclose(base[k], v, atol=1e-5, rtol=0):
        raise RuntimeError(f'Baseline mismatch for {k}: reproduced {base[k]:.8f}, original {v:.8f}. Check original training script and preprocessing before interpreting importance.')

rng = np.random.default_rng(SEED)
rows = []
for feature in FEATURES:
    for repeat in range(1, REPEATS + 1):
        perm = test[FEATURES].copy()
        perm[feature] = rng.permutation(perm[feature].to_numpy())
        m = metrics(perm)
        rows.append({'feature':feature, 'repeat':repeat,
                     **{f'permuted_{k}':m[k] for k in base},
                     **{f'drop_{k}':base[k]-m[k] for k in base}})
    print(f'Finished {feature}', flush=True)
raw = pd.DataFrame(rows)
raw.to_csv(OUT / 'rq6_coxph_permutation_repeats.csv', index=False)
summary = raw.groupby('feature', as_index=False).agg(**{
    f'{k}_{stat}': (f'drop_{k}', stat)
    for k in base for stat in ['mean','std']
})
summary = summary.sort_values('C_index_mean', ascending=False)
summary.to_csv(OUT / 'rq6_coxph_feature_importance.csv', index=False)
pd.DataFrame([base]).to_csv(OUT / 'rq6_coxph_reproduced_baseline.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(19, 7))
for ax, metric in zip(axes, base):
    ordered = summary.sort_values(f'{metric}_mean', ascending=True)
    ax.barh(ordered['feature'], ordered[f'{metric}_mean'],
            xerr=ordered[f'{metric}_std'], capsize=2)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'{metric} decrease after permutation')
    ax.set_xlabel('Mean decrease (± SD across 30 shuffles)')
fig.suptitle('RQ6: CoxPH permutation feature importance')
fig.tight_layout()
fig.savefig(OUT / 'rq6_coxph_importance_bar.png', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
order = summary['feature'].tolist()
for metric in base:
    ax.plot(range(len(order)), summary[f'{metric}_mean'], marker='o', label=metric)
ax.set_xticks(range(len(order)), order, rotation=50, ha='right')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Mean performance decrease')
ax.set_title('RQ6: Importance across evaluation metrics (ordered by C-index)')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'rq6_coxph_importance_line.png', dpi=220, bbox_inches='tight')
plt.close(fig)
print('\nSaved results to:', OUT.resolve())
print(summary.to_string(index=False))

