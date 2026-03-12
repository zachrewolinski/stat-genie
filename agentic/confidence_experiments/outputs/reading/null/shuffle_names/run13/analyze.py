import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Map columns based on metadata descriptions
# reader_view indicator is in 'language' (0/1)
# dyslexia binary is in 'correct_rate' (0/1) per metadata
# reading speed is assumed to be 'running_time' (only remaining metric without description)

# Defensive: drop rows with missing key fields
cols = ['language', 'correct_rate', 'running_time', 'device']
df = _df[cols].dropna().copy()

# Define dyslexia groups
# correct_rate: 1 = dyslexia, 0 = no dyslexia
# device: 0,1,2 for severity. We'll use correct_rate for main analysis and device for sensitivity.

# Main analysis: dyslexia == 1
main = df[df['correct_rate'] == 1]

# Compare reader_view (language==1) vs no reader_view (language==0)
rv = main[main['language'] == 1]['running_time']
no_rv = main[main['language'] == 0]['running_time']

# Descriptive stats

def summarize(series):
    return {
        'n': int(series.shape[0]),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std(ddof=1)),
    }

summary = {
    'rv': summarize(rv),
    'no_rv': summarize(no_rv),
}

# Welch t-test (unequal variances)
# If either group too small, result will be nan.
if rv.shape[0] >= 2 and no_rv.shape[0] >= 2:
    t_stat, p_val = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Nonparametric test (Mann-Whitney U)
if rv.shape[0] >= 2 and no_rv.shape[0] >= 2:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv, no_rv, alternative='two-sided')
    except Exception:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Effect size: Cohen's d

def cohens_d(a, b):
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.var(ddof=1)
    s2 = b.var(ddof=1)
    n1 = len(a)
    n2 = len(b)
    sp = ((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2)
    if sp <= 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(sp)

cohen_d = cohens_d(rv, no_rv)

# Sensitivity: use device dyslexia status (1 or 2) as dyslexic
sens = df[df['device'].isin([1,2])]
rv_s = sens[sens['language'] == 1]['running_time']
no_s = sens[sens['language'] == 0]['running_time']

summary_sens = {
    'rv': summarize(rv_s),
    'no_rv': summarize(no_s),
}

if rv_s.shape[0] >= 2 and no_s.shape[0] >= 2:
    t_stat_s, p_val_s = stats.ttest_ind(rv_s, no_s, equal_var=False, nan_policy='omit')
    try:
        u_stat_s, p_u_s = stats.mannwhitneyu(rv_s, no_s, alternative='two-sided')
    except Exception:
        u_stat_s, p_u_s = np.nan, np.nan
    cohen_d_s = cohens_d(rv_s, no_s)
else:
    t_stat_s, p_val_s = np.nan, np.nan
    u_stat_s, p_u_s = np.nan, np.nan
    cohen_d_s = np.nan

# Output results for inspection
print('Main dyslexia (correct_rate==1) summary:', summary)
print('Welch t-test: t=', t_stat, 'p=', p_val)
print('Mann-Whitney U: U=', u_stat, 'p=', p_u)
print('Cohen d:', cohen_d)

print('\nSensitivity (device in [1,2]) summary:', summary_sens)
print('Welch t-test: t=', t_stat_s, 'p=', p_val_s)
print('Mann-Whitney U: U=', u_stat_s, 'p=', p_u_s)
print('Cohen d:', cohen_d_s)
