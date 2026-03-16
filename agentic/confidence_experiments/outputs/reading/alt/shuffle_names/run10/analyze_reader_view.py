import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')

# Map columns to true meanings (inferred)
# reader_view flag appears in 'language' (0/1)
# reading speed appears in 'running_time'
# dyslexia status appears in 'device' (0 no, 1 dyslexia, 2 severe)
# participant id appears in 'speed'

# Basic cleaning
rv = df['language']
speed = df['running_time']
dys = df['device']

# Use dyslexic participants: dyslexia status 1 or 2
mask_dys = dys.isin([1.0, 2.0])

# Remove missing values for key vars
mask = mask_dys & rv.notna() & speed.notna()
sub = df.loc[mask, ['speed','language','running_time','device']].copy()

print('dyslexic rows', sub.shape[0])
print('dyslexic participants', sub['speed'].nunique())
print('reader_view counts', sub['language'].value_counts())

# Check whether reader_view varies within participant
rv_per_participant = sub.groupby('speed')['language'].nunique()
print('participants with both reader_view conditions', (rv_per_participant > 1).sum())
print('participants with single condition', (rv_per_participant == 1).sum())

# Overall comparison (rows)
rv0 = sub.loc[sub['language'] == 0, 'running_time']
rv1 = sub.loc[sub['language'] == 1, 'running_time']
print('mean speed rv0', rv0.mean(), 'rv1', rv1.mean())
print('median speed rv0', rv0.median(), 'rv1', rv1.median())

# Welch t-test
res_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('welch t-test', res_t)

# Mann-Whitney U test
res_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
print('mannwhitney', res_u)

# Effect size (Cohen's d)
mean_diff = rv1.mean() - rv0.mean()
# pooled SD (weighted)
var1 = rv1.var(ddof=1)
var0 = rv0.var(ddof=1)
n1, n0 = rv1.shape[0], rv0.shape[0]
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = mean_diff / pooled_sd
print('mean diff', mean_diff, 'cohen d', cohen_d)

# Participant-level aggregate (average speed per participant per condition)
# If participants have single condition, this reduces to between-subjects comparison
agg = sub.groupby(['speed','language'])['running_time'].mean().reset_index()
rv0_p = agg.loc[agg['language'] == 0, 'running_time']
rv1_p = agg.loc[agg['language'] == 1, 'running_time']
print('\nParticipant-level aggregates:')
print('n participants rv0', rv0_p.shape[0], 'rv1', rv1_p.shape[0])
print('mean speed rv0', rv0_p.mean(), 'rv1', rv1_p.mean())
print('median speed rv0', rv0_p.median(), 'rv1', rv1_p.median())
res_t_p = stats.ttest_ind(rv1_p, rv0_p, equal_var=False, nan_policy='omit')
print('welch t-test participants', res_t_p)

# Effect size for participant-level
var1p = rv1_p.var(ddof=1)
var0p = rv0_p.var(ddof=1)
n1p, n0p = rv1_p.shape[0], rv0_p.shape[0]
pooled_sd_p = np.sqrt(((n1p-1)*var1p + (n0p-1)*var0p) / (n1p+n0p-2))
cohen_d_p = (rv1_p.mean() - rv0_p.mean()) / pooled_sd_p
print('mean diff participants', rv1_p.mean() - rv0_p.mean(), 'cohen d', cohen_d_p)

# Paired comparison (within-participant)
paired = agg.pivot(index='speed', columns='language', values='running_time')
paired = paired.dropna()
diff = paired[1] - paired[0]
res_t_pair = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
print('\\nPaired test (within-participant):')
print('n pairs', diff.shape[0])
print('mean diff', diff.mean(), 'median diff', diff.median())
print('paired t-test', res_t_pair)
print('paired effect size (dz)', diff.mean() / diff.std(ddof=1))
