import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# define dyslexia group
# use dyslexia_bin if available; else dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'] > 0].copy()

# ensure reader_view and speed numeric
for col in ['reader_view', 'speed']:
    dys_df[col] = pd.to_numeric(dys_df[col], errors='coerce')

dys_df = dys_df.dropna(subset=['reader_view', 'speed'])

# split
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']

print('dyslexia n total:', len(dys_df))
print('n reader_view=1:', len(rv1), 'n reader_view=0:', len(rv0))

# descriptive stats
for label, series in [('rv1', rv1), ('rv0', rv0)]:
    print(label, 'mean', series.mean(), 'median', series.median(), 'std', series.std())

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('Welch t-test:', welch)

# log transform (add small epsilon)
eps = 1e-6
rv1_log = np.log(rv1 + eps)
rv0_log = np.log(rv0 + eps)
welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')
print('Welch t-test log:', welch_log)

# Mann-Whitney U
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U:', mwu)
except Exception as e:
    print('Mann-Whitney U error:', e)

# effect size Cohen's d (Hedges g)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)

n1, n0 = len(rv1), len(rv0)

# pooled SD for Hedges g
sp = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = (mean1 - mean0) / sp if sp != 0 else np.nan
# Hedges g correction
J = 1 - (3/(4*(n1+n0)-9))
hedges_g = cohen_d * J
print('Cohen d:', cohen_d, 'Hedges g:', hedges_g)

# relative difference
rel_diff = (mean1 - mean0) / mean0 if mean0 != 0 else np.nan
print('Relative mean difference:', rel_diff)

# bootstrap CI for mean difference
rng = np.random.default_rng(0)
boot_diffs = []
if n1 > 1 and n0 > 1:
    for _ in range(5000):
        samp1 = rng.choice(rv1.values, size=n1, replace=True)
        samp0 = rng.choice(rv0.values, size=n0, replace=True)
        boot_diffs.append(samp1.mean() - samp0.mean())
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    print('Bootstrap 95% CI mean diff:', ci_low, ci_high)

# also compare using adjusted_running_time/num_words maybe to compute speed if needed
# check if speed is derived; ensure we didn't need to compute words per minute
print('Columns:', list(df.columns))

