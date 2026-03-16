import pandas as pd
import numpy as np
from scipy import stats

# Load data
DF = pd.read_csv('reading.csv')

# Compute reading speed (words per minute) using reading time minus scrolling
# Avoid nonpositive or missing values
DF = DF.copy()
DF['wpm'] = DF['feature7'] * 60000 / DF['feature5']
DF = DF.replace([np.inf, -np.inf], np.nan)

# Filter dyslexia individuals
DF_dys = DF[DF['feature17'] == 1].copy()

# Reader View on/off
on = DF_dys[DF_dys['feature3'] == 1]['wpm'].dropna()
off = DF_dys[DF_dys['feature3'] == 0]['wpm'].dropna()

print('n dyslexia total', len(DF_dys))
print('n on', len(on), 'n off', len(off))
print('mean on', on.mean(), 'mean off', off.mean())
print('median on', on.median(), 'median off', off.median())

# Robust statistics: log-transform for t-test (Welch)
log_on = np.log(on)
log_off = np.log(off)

t_stat, p_val = stats.ttest_ind(log_on, log_off, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (non-parametric)
# Use two-sided test
u_stat, p_u = stats.mannwhitneyu(on, off, alternative='two-sided')

# Effect size: difference in means (raw) and Cohen's d on log scale
mean_diff = on.mean() - off.mean()
median_diff = on.median() - off.median()

# Cohen's d on log scale
sd_pooled = np.sqrt(((log_on.var(ddof=1)) + (log_off.var(ddof=1))) / 2)
cohen_d = (log_on.mean() - log_off.mean()) / sd_pooled if sd_pooled > 0 else np.nan

# Ratio of geometric means (exp of mean diff on log scale)
geom_ratio = np.exp(log_on.mean() - log_off.mean())

print('Welch t-test on log(wpm): t=', t_stat, 'p=', p_val)
print('Mann-Whitney U: U=', u_stat, 'p=', p_u)
print('mean diff (wpm):', mean_diff)
print('median diff (wpm):', median_diff)
print('Cohen d (log):', cohen_d)
print('Geom mean ratio (on/off):', geom_ratio)

# Trim extreme outliers (top/bottom 1%) to check robustness
low, high = DF_dys['wpm'].quantile([0.01, 0.99])
trim = DF_dys[(DF_dys['wpm'] >= low) & (DF_dys['wpm'] <= high)]

on_t = trim[trim['feature3'] == 1]['wpm']
off_t = trim[trim['feature3'] == 0]['wpm']
log_on_t = np.log(on_t)
log_off_t = np.log(off_t)

print('Trimmed 1% mean on', on_t.mean(), 'mean off', off_t.mean())
print('Trimmed 1% median on', on_t.median(), 'median off', off_t.median())


t_stat_t, p_val_t = stats.ttest_ind(log_on_t, log_off_t, equal_var=False, nan_policy='omit')

u_stat_t, p_u_t = stats.mannwhitneyu(on_t, off_t, alternative='two-sided')

sd_pooled_t = np.sqrt(((log_on_t.var(ddof=1)) + (log_off_t.var(ddof=1))) / 2)
cohen_d_t = (log_on_t.mean() - log_off_t.mean()) / sd_pooled_t if sd_pooled_t > 0 else np.nan
geom_ratio_t = np.exp(log_on_t.mean() - log_off_t.mean())

print('Trimmed Welch t-test log(wpm): t=', t_stat_t, 'p=', p_val_t)
print('Trimmed Mann-Whitney U: U=', u_stat_t, 'p=', p_u_t)
print('Trimmed Cohen d (log):', cohen_d_t)
print('Trimmed geom ratio:', geom_ratio_t)

# Also check reading speed based on total time (feature4) as sensitivity
DF['wpm_total'] = DF['feature7'] * 60000 / DF['feature4']
DF_dys2 = DF[DF['feature17'] == 1]

on2 = DF_dys2[DF_dys2['feature3'] == 1]['wpm_total']
off2 = DF_dys2[DF_dys2['feature3'] == 0]['wpm_total']

log_on2 = np.log(on2)
log_off2 = np.log(off2)

print('Total time wpm mean on', on2.mean(), 'mean off', off2.mean())
print('Total time wpm median on', on2.median(), 'median off', off2.median())
print('Total time log Welch p', stats.ttest_ind(log_on2, log_off2, equal_var=False, nan_policy='omit').pvalue)
print('Total time MW p', stats.mannwhitneyu(on2, off2, alternative='two-sided').pvalue)
