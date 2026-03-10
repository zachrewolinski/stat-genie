import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Compute reading speed (words per minute) using reading time minus scrolling (feature5)
# feature5 is milliseconds, feature7 is word count
_df = _df.copy()
_df['reading_time_min'] = _df['feature5'] / 60000.0
_df['wpm'] = _df['feature7'] / _df['reading_time_min']

# Filter invalids
_df = _df.replace([np.inf, -np.inf], np.nan)
_df = _df[_df['reading_time_min'] > 0]
_df = _df[_df['feature7'] > 0]

# Keep rows with dyslexia indicator and reader view indicator
_df = _df[_df['feature17'].notna() & _df['feature3'].notna()]

# Subset to dyslexic participants
_dys = _df[_df['feature17'] == 1]

# Group by reader view
rv_on = _dys[_dys['feature3'] == 1]['wpm']
rv_off = _dys[_dys['feature3'] == 0]['wpm']

# Remove extreme outliers using 1st/99th percentiles (winsorize by trimming)
if len(_dys) > 0:
    lower = _dys['wpm'].quantile(0.01)
    upper = _dys['wpm'].quantile(0.99)
    rv_on_trim = rv_on[(rv_on >= lower) & (rv_on <= upper)]
    rv_off_trim = rv_off[(rv_off >= lower) & (rv_off <= upper)]
else:
    rv_on_trim = rv_on
    rv_off_trim = rv_off

# Summary stats
summary = {
    'n_total_dys': int(len(_dys)),
    'n_on': int(len(rv_on_trim)),
    'n_off': int(len(rv_off_trim)),
    'mean_on': float(rv_on_trim.mean()),
    'mean_off': float(rv_off_trim.mean()),
    'median_on': float(rv_on_trim.median()),
    'median_off': float(rv_off_trim.median()),
}

# Log transform for skew
log_on = np.log(rv_on_trim)
log_off = np.log(rv_off_trim)

# Welch t-test on log speeds
if len(log_on) > 1 and len(log_off) > 1:
    t_stat, p_val = stats.ttest_ind(log_on, log_off, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Mann-Whitney U test (two-sided)
if len(rv_on_trim) > 0 and len(rv_off_trim) > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv_on_trim, rv_off_trim, alternative='two-sided')
    except ValueError:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Effect size (Cohen's d) on log speeds
if len(log_on) > 1 and len(log_off) > 1:
    mean_on = np.nanmean(log_on)
    mean_off = np.nanmean(log_off)
    var_on = np.nanvar(log_on, ddof=1)
    var_off = np.nanvar(log_off, ddof=1)
    n_on = np.sum(~np.isnan(log_on))
    n_off = np.sum(~np.isnan(log_off))
    pooled = np.sqrt(((n_on - 1) * var_on + (n_off - 1) * var_off) / (n_on + n_off - 2))
    cohen_d = (mean_on - mean_off) / pooled if pooled > 0 else np.nan
else:
    cohen_d = np.nan

print('Summary:', summary)
print('Welch t-test (log wpm): t=', t_stat, 'p=', p_val)
print('Mann-Whitney U: U=', u_stat, 'p=', p_u)
print('Cohen d (log wpm):', cohen_d)

# Also compute percent difference in medians
if summary['median_off'] > 0:
    pct_diff_median = (summary['median_on'] - summary['median_off']) / summary['median_off'] * 100.0
else:
    pct_diff_median = np.nan
print('Median % difference (on vs off):', pct_diff_median)

