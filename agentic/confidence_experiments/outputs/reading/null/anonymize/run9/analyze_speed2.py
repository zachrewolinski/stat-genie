import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

# Use feature20 as provided reading speed metric (unknown label)
df = df.copy()

sub = df[df['feature17'] == 1].copy()
sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['feature20'])

rv_on = sub[sub['feature3'] == 1]['feature20']
rv_off = sub[sub['feature3'] == 0]['feature20']

print('feature20 on desc', rv_on.describe())
print('feature20 off desc', rv_off.describe())

welch = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
print('Welch t-test', welch)

# Hedges g
n1, n0 = rv_on.shape[0], rv_off.shape[0]
mean1, mean0 = rv_on.mean(), rv_off.mean()
var1, var0 = rv_on.var(ddof=1), rv_off.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1 + n0 - 2))
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
J = 1 - (3 / (4*(n1+n0)-9)) if (n1+n0) > 2 else 1
hedges_g = cohen_d * J
print('Hedges g', hedges_g)

# Paired analysis
agg = sub.groupby(['feature1','feature3'])['feature20'].mean().reset_index()
wide = agg.pivot(index='feature1', columns='feature3', values='feature20')
paired = wide.dropna()
print('paired participants', paired.shape[0])
if paired.shape[0] > 1:
    ttest_paired = stats.ttest_rel(paired[1], paired[0])
    print('paired t-test', ttest_paired)
    diff = paired[1] - paired[0]
    print('paired mean diff', diff.mean())
    d_paired = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    print('paired cohen d', d_paired)

