import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map variables based on value patterns
reader_view = df['language']  # 0/1 balanced
reading_speed = df['running_time']  # inferred WPM
# Dyslexia status inferred from 'device' (0 none, 1 dyslexia, 2 severe)

dys = df['device']

# subset dyslexic (1 or 2)
mask = dys.isin([1,2]) & reader_view.notna() & reading_speed.notna()
sub = df.loc[mask].copy()

sub['reader_view'] = reader_view[mask]
sub['speed'] = reading_speed[mask]

# groups
rv_on = sub[sub['reader_view'] == 1]['speed']
rv_off = sub[sub['reader_view'] == 0]['speed']

print('dyslexic sample size total', len(sub))
print('reader_view on', len(rv_on), 'off', len(rv_off))

# descriptive stats
for name, series in [('on', rv_on), ('off', rv_off)]:
    print(name, 'mean', series.mean(), 'median', series.median(), 'std', series.std(), 'min', series.min(), 'max', series.max())

# Welch t-test
welch = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
print('Welch t-test', welch)

# Mann-Whitney U
try:
    mwu = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
    print('Mann-Whitney U', mwu)
except Exception as e:
    print('Mann-Whitney failed', e)

# effect size (Cohen d)
mean_diff = rv_on.mean() - rv_off.mean()
# pooled sd for Welch? use sqrt((s1^2+s2^2)/2)
pooled_sd = np.sqrt((rv_on.var(ddof=1) + rv_off.var(ddof=1)) / 2)
cohen_d = mean_diff / pooled_sd
print('mean_diff', mean_diff, 'cohen_d', cohen_d)

# Also log-transform to reduce skew
rv_on_log = np.log1p(rv_on)
rv_off_log = np.log1p(rv_off)
welch_log = stats.ttest_ind(rv_on_log, rv_off_log, equal_var=False, nan_policy='omit')
mean_diff_log = rv_on_log.mean() - rv_off_log.mean()
pooled_sd_log = np.sqrt((rv_on_log.var(ddof=1) + rv_off_log.var(ddof=1)) / 2)
cohen_d_log = mean_diff_log / pooled_sd_log
print('Welch log t-test', welch_log)
print('mean_diff_log', mean_diff_log, 'cohen_d_log', cohen_d_log)

# bootstrapped CI for mean difference
rng = np.random.default_rng(42)
boot = []
for _ in range(5000):
    boot_on = rng.choice(rv_on.values, size=len(rv_on), replace=True)
    boot_off = rng.choice(rv_off.values, size=len(rv_off), replace=True)
    boot.append(boot_on.mean() - boot_off.mean())
boot = np.array(boot)
ci = np.percentile(boot, [2.5, 97.5])
print('boot mean diff 95% CI', ci)

