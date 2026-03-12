import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

# Focus on dyslexic participants (dyslexia_bin==1). If missing, use dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

print('rows total', len(df), 'dys rows', len(dys))
print('unique uuids total', df['uuid'].nunique(), 'dys uuids', dys['uuid'].nunique())
print('reader_view counts dys', dys['reader_view'].value_counts())

# describe speed by reader_view
summary = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print(summary)

# check per-uuid within-subject availability
pivot = dys.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
print('participants with both conditions', pivot.dropna().shape[0])
print(pivot.head())

# independent t-test (Welch)
rv1 = dys[dys['reader_view'] == 1]['speed'].dropna()
rv0 = dys[dys['reader_view'] == 0]['speed'].dropna()
welch = stats.ttest_ind(rv1, rv0, equal_var=False)
print('welch t-test', welch)

# Mann-Whitney U
mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
print('mann-whitney', mw)

# Paired test for participants with both conditions
paired = pivot.dropna()
if len(paired) > 0:
    t_paired = stats.ttest_rel(paired[1], paired[0])
    print('paired t-test', t_paired)
    w_paired = stats.wilcoxon(paired[1], paired[0])
    print('wilcoxon', w_paired)
    # effect size d for paired differences
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1)
    print('paired mean diff', diff.mean(), 'std diff', diff.std(ddof=1), 'd', d)

# effect size (Cohen d) for independent
# use pooled SD for d
n1, n0 = len(rv1), len(rv0)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0)/(n1+n0-2))
cohen_d = (mean1-mean0)/pooled
print('independent d', cohen_d)

# log speed to reduce skew
log1 = np.log1p(rv1)
log0 = np.log1p(rv0)
welch_log = stats.ttest_ind(log1, log0, equal_var=False)
print('welch log', welch_log)

# paired log test
if len(paired) > 0:
    log_paired = np.log1p(paired)
    t_paired_log = stats.ttest_rel(log_paired[1], log_paired[0])
    print('paired t log', t_paired_log)
