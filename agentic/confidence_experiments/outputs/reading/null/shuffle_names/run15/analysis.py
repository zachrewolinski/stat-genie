import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Map columns based on value patterns (see AGENTS instructions)
reader_view = _df['language']  # binary 0/1, balanced
# Dyslexia status encoded in 'device' (0=no, 1=dyslexia, 2=severe)
dyslexia_status = _df['device']
# Reading speed likely in 'running_time' (continuous; median ~287)
reading_speed = _df['running_time']

# Subset to dyslexic participants (device > 0)
mask = dyslexia_status.notna() & reader_view.notna() & reading_speed.notna()
_df2 = _df.loc[mask].copy()
_df2 = _df2[_df2['device'] > 0]

# Group by reader_view (0/1)
rv0 = _df2[_df2['language'] == 0]['running_time']
rv1 = _df2[_df2['language'] == 1]['running_time']

# Basic stats
stats_summary = {
    'n_total_dyslexic': len(_df2),
    'n_reader_view_0': len(rv0),
    'n_reader_view_1': len(rv1),
    'mean_0': rv0.mean(),
    'mean_1': rv1.mean(),
    'median_0': rv0.median(),
    'median_1': rv1.median(),
}

# Effect sizes
# Cohen's d (using pooled std)
mean0, mean1 = rv0.mean(), rv1.mean()
std0, std1 = rv0.std(ddof=1), rv1.std(ddof=1)
pooled_std = np.sqrt(((len(rv0)-1)*std0**2 + (len(rv1)-1)*std1**2) / (len(rv0)+len(rv1)-2))
cohens_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else np.nan

# Mann-Whitney U test (nonparametric)
# Use two-sided test, handle ties with alternative method if needed
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    mwu_stat, mwu_p = mwu.statistic, mwu.pvalue
except Exception:
    mwu_stat, mwu_p = np.nan, np.nan

# Welch's t-test
try:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
except Exception:
    t_stat, t_p = np.nan, np.nan

# Cliff's delta (robust effect size)
# Compute via rank-based formula
# delta = (2*U)/(n1*n0) - 1
n1, n0 = len(rv1), len(rv0)
if n1 > 0 and n0 > 0 and not np.isnan(mwu_stat):
    cliffs_delta = (2 * mwu_stat) / (n1 * n0) - 1
else:
    cliffs_delta = np.nan

results = {
    'stats': stats_summary,
    'cohens_d': cohens_d,
    't_p': t_p,
    'mwu_p': mwu_p,
    'cliffs_delta': cliffs_delta,
}

print(results)
