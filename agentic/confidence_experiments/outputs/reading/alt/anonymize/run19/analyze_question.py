import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Identify dyslexia status
# feature17: 1 dyslexia, 0 no dyslexia
# feature12: 0 no, 1 dyslexia, 2 severe dyslexia

# use feature17 when available, fallback to feature12

df['dyslexic'] = df['feature17']
# where feature17 missing, use feature12 > 0
mask_missing = df['dyslexic'].isna()
df.loc[mask_missing, 'dyslexic'] = (df.loc[mask_missing, 'feature12'] > 0).astype(float)

# subset dyslexic individuals
sub = df[df['dyslexic'] == 1].copy()

# reader view indicator
sub['reader_view'] = sub['feature3']

# reading speed (wpm)
sub['reading_speed'] = sub['feature20']

# Summary stats by condition
summary = sub.groupby('reader_view')['reading_speed'].agg(['count', 'mean', 'median', 'std'])

# participant-level means for paired comparison
# feature1 is participant ID
pivot = sub.pivot_table(index='feature1', columns='reader_view', values='reading_speed', aggfunc='mean')

# keep participants with both conditions
paired = pivot.dropna()

# compute paired t-test
if len(paired) > 1:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    # Cohen's d for paired samples
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1)
else:
    t_stat, p_val, d = np.nan, np.nan, np.nan

# simple independent comparison as sensitivity
rv1 = sub[sub['reader_view'] == 1]['reading_speed']
rv0 = sub[sub['reader_view'] == 0]['reading_speed']

if len(rv1) > 1 and len(rv0) > 1:
    t_stat_ind, p_val_ind = stats.ttest_ind(rv1, rv0, equal_var=False)
else:
    t_stat_ind, p_val_ind = np.nan, np.nan

# also check non-parametric Wilcoxon on paired data
if len(paired) > 1:
    try:
        w_stat, p_val_w = stats.wilcoxon(paired[1], paired[0])
    except Exception:
        w_stat, p_val_w = np.nan, np.nan
else:
    w_stat, p_val_w = np.nan, np.nan

print('Dyslexic subset size (rows):', len(sub))
print('Unique participants (dyslexic):', sub['feature1'].nunique())
print('\nSummary by reader_view:')
print(summary)

print('\nPaired participants with both conditions:', len(paired))
print('Paired t-test: t=%.3f, p=%.6f, Cohen d=%.3f' % (t_stat, p_val, d))
print('Wilcoxon: stat=%s, p=%s' % (w_stat, p_val_w))
print('\nIndependent t-test (Welch): t=%.3f, p=%.6f' % (t_stat_ind, p_val_ind))

# effect size in raw units
if len(paired) > 0:
    mean_diff = (paired[1]-paired[0]).mean()
    print('Mean within-subject difference (reader_view - no):', mean_diff)

# Provide some quantiles for robustness
print('\nQuantiles of reading speed by condition:')
for rv in [0, 1]:
    vals = sub[sub['reader_view'] == rv]['reading_speed']
    q = vals.quantile([0.25, 0.5, 0.75])
    print('reader_view', rv, 'n', len(vals), 'q25/q50/q75', q.to_dict())
