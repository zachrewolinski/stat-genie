import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Basic cleaning
# Ensure numeric columns
for col in ['reader_view','speed','dyslexia','dyslexia_bin','retake_trial']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Filter dyslexia individuals (dyslexia_bin == 1)
_df_dys = _df[_df['dyslexia_bin'] == 1].copy()

# Optionally drop retake trials to avoid practice effects
_df_dys_no_retake = _df_dys[_df_dys['retake_trial'] == 0].copy()


def summarize_condition(df):
    counts = df.groupby('reader_view')['speed'].agg(['count','mean','median','std']).reset_index()
    return counts


def paired_analysis(df):
    # Compute per-participant mean speed by reader_view
    pivot = df.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
    # Columns 0 and 1 represent no reader view / reader view
    if 0 not in pivot.columns or 1 not in pivot.columns:
        return None
    paired = pivot.dropna(subset=[0,1])
    if paired.empty:
        return None
    diff = paired[1] - paired[0]
    # Paired t-test
    t_res = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    # Wilcoxon signed-rank (if at least 10 pairs and no all-zero diffs)
    wilcoxon_res = None
    try:
        if len(diff) >= 10 and not np.allclose(diff, 0):
            wilcoxon_res = stats.wilcoxon(diff)
    except Exception:
        wilcoxon_res = None
    # Cohen's d for paired samples: mean(diff) / std(diff)
    diff_std = diff.std(ddof=1)
    cohen_d = diff.mean() / diff_std if diff_std and not np.isnan(diff_std) else np.nan
    return {
        'n_pairs': int(len(paired)),
        'mean_diff': float(diff.mean()),
        'median_diff': float(diff.median()),
        't_stat': float(t_res.statistic),
        't_p': float(t_res.pvalue),
        'cohen_d': float(cohen_d),
        'wilcoxon_stat': float(wilcoxon_res.statistic) if wilcoxon_res is not None else None,
        'wilcoxon_p': float(wilcoxon_res.pvalue) if wilcoxon_res is not None else None,
    }


def independent_analysis(df):
    # Use per-participant means to reduce within-subject correlation
    per_participant = df.groupby(['uuid','reader_view'])['speed'].mean().reset_index()
    group0 = per_participant[per_participant['reader_view'] == 0]['speed']
    group1 = per_participant[per_participant['reader_view'] == 1]['speed']
    if len(group0) < 2 or len(group1) < 2:
        return None
    t_res = stats.ttest_ind(group1, group0, equal_var=False, nan_policy='omit')
    # Mann-Whitney U test
    mw_res = stats.mannwhitneyu(group1, group0, alternative='two-sided')
    # Cohen's d (Hedges g) for independent
    n1, n0 = len(group1), len(group0)
    s1, s0 = group1.std(ddof=1), group0.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2)) if n1+n0-2 > 0 else np.nan
    d = (group1.mean() - group0.mean()) / pooled if pooled and not np.isnan(pooled) else np.nan
    return {
        'n1': int(n1),
        'n0': int(n0),
        'mean1': float(group1.mean()),
        'mean0': float(group0.mean()),
        't_stat': float(t_res.statistic),
        't_p': float(t_res.pvalue),
        'mw_stat': float(mw_res.statistic),
        'mw_p': float(mw_res.pvalue),
        'cohen_d': float(d),
    }


results = {
    'overall_counts': summarize_condition(_df_dys).to_dict(orient='records'),
    'overall_paired': paired_analysis(_df_dys),
    'overall_independent': independent_analysis(_df_dys),
    'no_retake_counts': summarize_condition(_df_dys_no_retake).to_dict(orient='records'),
    'no_retake_paired': paired_analysis(_df_dys_no_retake),
    'no_retake_independent': independent_analysis(_df_dys_no_retake),
    'n_rows_dys': int(len(_df_dys)),
    'n_participants_dys': int(_df_dys['uuid'].nunique()),
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
