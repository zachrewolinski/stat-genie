import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
csv_path = 'reading.csv'
info_path = 'info.json'

info = json.load(open(info_path))

df = pd.read_csv(csv_path)

# Basic cleaning: ensure numeric
for col in ['reader_view', 'speed', 'dyslexia_bin', 'dyslexia']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter to dyslexic individuals
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    # fallback: dyslexia >=1
    dys_df = df[df['dyslexia'].fillna(0) >= 1].copy()

# Drop missing speed or reader_view
dys_df = dys_df.dropna(subset=['speed', 'reader_view', 'uuid'])

# Descriptive stats by reader_view
group_stats = dys_df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Paired analysis: per participant mean speed by reader_view
pivot = dys_df.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')

# Expect reader_view values 0 and 1
paired = pivot.dropna(subset=[0, 1]) if 0 in pivot.columns and 1 in pivot.columns else pd.DataFrame()

results = {}

# Independent Welch t-test as fallback / supplemental
if 0 in group_stats.index and 1 in group_stats.index:
    speed0 = dys_df[dys_df['reader_view'] == 0]['speed']
    speed1 = dys_df[dys_df['reader_view'] == 1]['speed']
    t_stat, p_val = stats.ttest_ind(speed1, speed0, equal_var=False, nan_policy='omit')
    # Cohen's d (independent, pooled SD)
    n0, n1 = speed0.dropna().shape[0], speed1.dropna().shape[0]
    s0, s1 = speed0.dropna().std(ddof=1), speed1.dropna().std(ddof=1)
    s_pooled = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2)) if (n0+n1-2) > 0 else np.nan
    cohend = (speed1.mean() - speed0.mean()) / s_pooled if s_pooled and not np.isnan(s_pooled) else np.nan
    results['welch_ttest'] = {
        't_stat': t_stat,
        'p_value': p_val,
        'mean_diff': speed1.mean() - speed0.mean(),
        'cohen_d': cohend,
        'n0': n0,
        'n1': n1
    }

# Paired t-test on within-participant means
if not paired.empty:
    diff = paired[1] - paired[0]
    t_stat_p, p_val_p = stats.ttest_1samp(diff, 0.0, nan_policy='omit')
    # 95% CI
    n = diff.dropna().shape[0]
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n) if n > 0 else np.nan
    t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
    ci_low = mean_diff - t_crit * se if n > 1 else np.nan
    ci_high = mean_diff + t_crit * se if n > 1 else np.nan
    # Cohen's dz
    cohendz = mean_diff / sd_diff if sd_diff and not np.isnan(sd_diff) else np.nan
    # Wilcoxon signed-rank (non-param)
    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    except Exception:
        w_stat, w_p = (np.nan, np.nan)

    results['paired_ttest'] = {
        'n_pairs': n,
        'mean_diff': mean_diff,
        'median_diff': diff.median(),
        't_stat': t_stat_p,
        'p_value': p_val_p,
        'ci_95': (ci_low, ci_high),
        'cohen_dz': cohendz,
        'wilcoxon_p': w_p
    }

# Save results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump({
        'group_stats': group_stats.to_dict(),
        'results': results,
        'n_dyslexic_rows': int(dys_df.shape[0]),
        'n_unique_dyslexic': int(dys_df['uuid'].nunique()),
        'n_pairs': int(paired.shape[0]) if not paired.empty else 0
    }, f, indent=2)

print(json.dumps({
    'group_stats': group_stats.to_dict(),
    'results': results,
    'n_dyslexic_rows': int(dys_df.shape[0]),
    'n_unique_dyslexic': int(dys_df['uuid'].nunique()),
    'n_pairs': int(paired.shape[0]) if not paired.empty else 0
}, indent=2))
