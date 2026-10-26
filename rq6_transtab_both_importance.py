import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import transtab

COMMON = ['prior_dx_count','prior_unique_dx_count','prior_visit_count','age_at_last_prior_visit','mean_codes_per_visit','max_codes_per_visit','mean_unique_codes_per_visit','prior_icd9_count','prior_icd10_count']
TEMPORAL = ['mean_intervisit_gap_days','max_intervisit_gap_days','last_intervisit_gap_days','last_visit_recency_days','history_span_days']
CONFIGS = [
    ('control', 'structured_features_365d_safe.parquet', COMMON, 'transtab_365d_control_ckpt', 'transtab_365d_control_results.csv'),
    ('timeaware', 'structured_features_timeaware_365d_safe.parquet', COMMON + TEMPORAL, 'timeaware_transtab_365d_ckpt', 'timeaware_transtab_365d_results.csv')
]
OUT = Path('rq6_transtab_outputs')
OUT.mkdir(exist_ok=True)
SPLITS = pd.read_parquet('patient_splits_timeaware_365d.parquet')
REPEATS = 30
all_summaries = []

for name, data_path, features, checkpoint, result_path in CONFIGS:
    print(f'\n=== TransTab {name} ===', flush=True)
    df = pd.read_parquet(data_path).merge(SPLITS, on='subject_id', how='inner', validate='one_to_one')
    train = df.loc[df.split == 'train'].copy()
    test = df.loc[df.split == 'test'].copy()
    scaler = StandardScaler()
    scaler.fit(train[features])
    X = pd.DataFrame(scaler.transform(test[features]), columns=features).astype(float).reset_index(drop=True)
    y = test['dn_within_365d'].to_numpy(dtype=int)
    assert len(y) == 1337 and int(y.sum()) == 103, (len(y), int(y.sum()))

    model = transtab.build_classifier(
        categorical_columns=[], numerical_columns=features, binary_columns=[],
        num_class=2, hidden_dim=128, num_layer=2, num_attention_head=8,
        hidden_dropout_prob=0.1, ffn_dim=256, activation='relu', device='cpu'
    )
    model.load('./' + checkpoint)
    model.eval()

    def predict(frame):
        scores = np.asarray(transtab.predict(model, frame, eval_batch_size=256)).reshape(-1)
        if len(scores) != len(y):
            raise RuntimeError(f'Unexpected prediction length: {len(scores)}')
        return scores

    baseline = predict(X)
    auc = roc_auc_score(y, baseline)
    ap = average_precision_score(y, baseline)
    expected = pd.read_csv(result_path).iloc[0]
    print(f'Baseline AUROC={auc:.6f}, AUPRC={ap:.6f}', flush=True)
    print(f'Original AUROC={float(expected.AUROC):.6f}, AUPRC={float(expected.AUPRC):.6f}', flush=True)
    if abs(auc - float(expected.AUROC)) > .001 or abs(ap - float(expected.AUPRC)) > .001:
        raise RuntimeError(f'{name} baseline mismatch: stopped before calculating importance')

    rng = np.random.default_rng(2026)
    rows = []
    for feature in features:
        for repeat in range(1, REPEATS + 1):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(X[feature].to_numpy())
            prob = predict(shuffled)
            rows.append(dict(model=name, feature=feature,
                             group='Temporal' if feature in TEMPORAL else 'Clinical history',
                             repeat=repeat, AUROC_drop=auc-roc_auc_score(y, prob),
                             AUPRC_drop=ap-average_precision_score(y, prob)))
        print('Finished:', feature, flush=True)

    repeats = pd.DataFrame(rows)
    repeats.to_csv(OUT / f'{name}_permutation_repeats.csv', index=False)
    summary = repeats.groupby(['model','feature','group'], as_index=False).agg(
        AUROC_mean=('AUROC_drop','mean'), AUROC_std=('AUROC_drop','std'),
        AUPRC_mean=('AUPRC_drop','mean'), AUPRC_std=('AUPRC_drop','std'))
    summary = summary.sort_values('AUROC_mean', ascending=False)
    summary.to_csv(OUT / f'{name}_feature_importance.csv', index=False)
    pd.DataFrame([dict(model=name, N_test=len(y), events=int(y.sum()), AUROC=auc, AUPRC=ap)]).to_csv(OUT / f'{name}_baseline.csv', index=False)
    all_summaries.append(summary)

    for metric in ['AUROC','AUPRC']:
        ordered = summary.sort_values(f'{metric}_mean')
        plt.figure(figsize=(11,7))
        plt.barh(ordered.feature, ordered[f'{metric}_mean'], xerr=ordered[f'{metric}_std'], capsize=3)
        plt.axvline(0, color='black', linewidth=.8)
        plt.xlabel(f'Mean {metric} reduction (30 permutations)')
        plt.title(f'TransTab {name}: {metric} permutation importance')
        plt.tight_layout()
        plt.savefig(OUT / f'{name}_{metric.lower()}_bar.png', dpi=180)
        plt.close()
    plt.figure(figsize=(12,6))
    pos = np.arange(len(summary))
    plt.plot(pos, summary.AUROC_mean, marker='o', label='AUROC')
    plt.plot(pos, summary.AUPRC_mean, marker='s', label='AUPRC')
    plt.xticks(pos, summary.feature, rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=.8)
