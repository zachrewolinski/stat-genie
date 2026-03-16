import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = 'reading.csv'

df = pd.read_csv(DATA_PATH)

# Ensure expected columns
required_cols = {'reader_view','speed','dyslexia_bin','uuid'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Focus on individuals with dyslexia

df_dys = df[df['dyslexia_bin'] == 1].copy()

# Basic counts
n_rows = len(df_dys)

# Group stats by reader_view
summary = df_dys.groupby('reader_view')['speed'].agg(['count','mean','median','std']).rename(index={0:'off',1:'on'})

# Check per-uuid paired availability
pivot = df_dys.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna()

paired_n = len(paired)

diff = paired[1] - paired[0]

# Paired t-test
if paired_n > 1:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
else:
    t_stat, p_val = np.nan, np.nan

# Wilcoxon signed-rank (non-parametric) if enough
if paired_n > 10:
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(paired[1], paired[0])
    except ValueError:
        p_wilcoxon = np.nan
else:
    p_wilcoxon = np.nan

# Effect size (Cohen's d for paired)
if paired_n > 1:
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
else:
    d = np.nan

# Also compute Welch t-test on all rows (ignores pairing) for robustness
speed_on = df_dys[df_dys['reader_view'] == 1]['speed']
speed_off = df_dys[df_dys['reader_view'] == 0]['speed']

if len(speed_on) > 1 and len(speed_off) > 1:
    t_welch, p_welch = stats.ttest_ind(speed_on, speed_off, equal_var=False)
else:
    t_welch, p_welch = np.nan, np.nan

result = {
    'n_rows_dyslexia': int(n_rows),
    'summary_by_reader_view': summary.reset_index().to_dict(orient='records'),
    'paired_n': int(paired_n),
    'paired_diff_mean': float(diff.mean()) if paired_n>0 else np.nan,
    'paired_diff_median': float(diff.median()) if paired_n>0 else np.nan,
    'paired_t_pvalue': float(p_val) if not np.isnan(p_val) else None,
    'paired_t_stat': float(t_stat) if not np.isnan(t_stat) else None,
    'wilcoxon_pvalue': float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
    'welch_pvalue': float(p_welch) if not np.isnan(p_welch) else None,
    'cohens_d_paired': float(d) if not np.isnan(d) else None,
    'speed_on_count': int(len(speed_on)),
    'speed_off_count': int(len(speed_off)),
    'speed_on_mean': float(speed_on.mean()) if len(speed_on)>0 else None,
    'speed_off_mean': float(speed_off.mean()) if len(speed_off)>0 else None,
    'speed_on_median': float(speed_on.median()) if len(speed_on)>0 else None,
    'speed_off_median': float(speed_off.median()) if len(speed_off)>0 else None,
}

with open('analysis_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
