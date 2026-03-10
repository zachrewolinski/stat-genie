import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Column mapping inferred from metadata/value patterns
reader_view_col = 'language'  # 0/1 indicates reader view on/off
reading_speed_col = 'running_time'  # numeric, behaves like reading speed
# Dyslexia status inferred as 0=no dyslexia, 1=dyslexia, 2=severe
# from distribution and metadata

dyslexia_col = 'device'

# Filter dyslexia participants (1 or 2) and non-missing relevant values
sub = df[df[dyslexia_col].isin([1.0, 2.0])].copy()
sub = sub[[reader_view_col, reading_speed_col]].dropna()

# Ensure binary reader_view
# It should be 0/1; keep as is

# Split groups
rv0 = sub[sub[reader_view_col] == 0][reading_speed_col]
rv1 = sub[sub[reader_view_col] == 1][reading_speed_col]

# Descriptive stats

def desc(series):
    return {
        'n': int(series.shape[0]),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
    }

summary = {
    'reader_view_0': desc(rv0),
    'reader_view_1': desc(rv1),
}

# Welch t-test (unequal variances)

t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric)
try:
    u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d

def cohens_d(a, b):
    a = a.dropna()
    b = b.dropna()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(pooled)

cohen_d = cohens_d(rv1, rv0)

# Log-transform for robustness (add small constant to avoid log(0))
log_rv0 = np.log(rv0 + 1e-6)
log_rv1 = np.log(rv1 + 1e-6)
log_t_stat, log_t_p = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')

result = {
    'summary': summary,
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mw_test': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    'cohens_d': float(cohen_d),
    'log_t_test': {'t_stat': float(log_t_stat), 'p_value': float(log_t_p)},
    'n_total': int(sub.shape[0])
}

print(result)
