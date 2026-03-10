import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Map columns
# reader_view indicator is 'language'
# dyslexia status is 'device' (0 no,1 dyslexia,2 severe)
# reading speed is 'running_time'

# Filter dyslexic individuals (device 1 or 2)
dys = df[df['device'].isin([1.0, 2.0])].copy()

# group by reader_view (language)
rv0 = dys[dys['language'] == 0]['running_time'].dropna()
rv1 = dys[dys['language'] == 1]['running_time'].dropna()

print('dyslexic sample sizes', len(rv0), len(rv1))

# Summary stats
for label, series in [('ReaderView=0', rv0), ('ReaderView=1', rv1)]:
    print(label, 'mean', series.mean(), 'median', series.median(), 'std', series.std())

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('Welch t-test', welch)

# Mann-Whitney U (two-sided)
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U', mw)
except Exception as e:
    print('Mann-Whitney error', e)

# Effect size: Cohen's d for unequal sizes
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)

n1, n0 = len(rv1), len(rv0)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
print('Cohen d', cohens_d)

# log-transform (handle zeros if any)
rv0_log = np.log1p(rv0)
rv1_log = np.log1p(rv1)
welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')
print('Welch t-test log1p', welch_log)

# Effect size on log scale
mean1_log, mean0_log = rv1_log.mean(), rv0_log.mean()
var1_log, var0_log = rv1_log.var(ddof=1), rv0_log.var(ddof=1)
pooled_sd_log = np.sqrt(((n1-1)*var1_log + (n0-1)*var0_log) / (n1+n0-2))
cohens_d_log = (mean1_log - mean0_log) / pooled_sd_log if pooled_sd_log > 0 else np.nan
print('Cohen d log', cohens_d_log)

# Bootstrap difference in medians
rng = np.random.default_rng(0)
B = 5000
med_diffs = []
for _ in range(B):
    s1 = rng.choice(rv1.values, size=n1, replace=True)
    s0 = rng.choice(rv0.values, size=n0, replace=True)
    med_diffs.append(np.median(s1) - np.median(s0))
med_diffs = np.array(med_diffs)
ci = np.percentile(med_diffs, [2.5, 97.5])
print('median diff', np.median(rv1) - np.median(rv0), 'bootstrap 95% CI', ci)

