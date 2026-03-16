import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Focus on dyslexic participants (dyslexia_bin==1)
# Some rows may have missing speed; drop
sub = df[df['dyslexia_bin'] == 1].copy()
sub = sub.dropna(subset=['speed','reader_view'])

# Basic group stats
stats_group = sub.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# Effect size: difference in means (rv=1 - rv=0)
mean1 = stats_group.loc[1, 'mean'] if 1 in stats_group.index else np.nan
mean0 = stats_group.loc[0, 'mean'] if 0 in stats_group.index else np.nan
mean_diff = mean1 - mean0

# Welch t-test
rv1 = sub[sub['reader_view']==1]['speed']
rv0 = sub[sub['reader_view']==0]['speed']

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mixed effects: random intercept per participant
# If uuid exists; use log speed for skewness
sub = sub.dropna(subset=['uuid'])
sub['log_speed'] = np.log(sub['speed'].clip(lower=1e-6))

# Mixed model (may fail if singular) -- try/except
mix_res = None
try:
    md = smf.mixedlm('log_speed ~ reader_view', sub, groups=sub['uuid'])
    mix_res = md.fit(reml=False, method='lbfgs')
except Exception as e:
    mix_res = e

# Also participant-level aggregation (mean per uuid per reader_view)
# then paired if both conditions present
agg = sub.groupby(['uuid','reader_view'])['speed'].mean().unstack()
paired = agg.dropna()
paired_diff = (paired[1] - paired[0])
paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit') if len(paired) > 1 else None

# Save summary to stdout
print('N dyslexic rows:', len(sub))
print('Group stats (reader_view):')
print(stats_group)
print('Mean diff (rv=1 - rv=0):', mean_diff)
print('Welch t-test:', ttest)
print('Paired N (both conditions):', len(paired))
if paired_t:
    print('Paired t-test:', paired_t)
else:
    print('Paired t-test: insufficient data')

print('MixedLM result:')
print(mix_res)
