import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Identify key columns based on observed data characteristics
# reader_view indicator is in `language` (0/1), dyslexia status in `dyslexia` (0/1/2), reading speed in `running_time`

# Filter dyslexic participants (dyslexia > 0)
dys = df[df['dyslexia'] > 0].copy()

# Basic group stats
rv0 = dys[dys['language'] == 0]['running_time']
rv1 = dys[dys['language'] == 1]['running_time']

# Independent group tests
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
mannwhit = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

# Effect size (Cohen's d for independent samples)
mean0, mean1 = rv0.mean(), rv1.mean()
std0, std1 = rv0.std(ddof=1), rv1.std(ddof=1)
# pooled SD (Welch-style weighting)
pooled = np.sqrt(((rv0.count()-1)*std0**2 + (rv1.count()-1)*std1**2) / (rv0.count()+rv1.count()-2))
cohen_d = (mean1 - mean0) / pooled if pooled > 0 else np.nan

# Paired analysis: per-participant averages
# Participant id appears in `speed` column (uuid-like), compute mean speed per participant per condition
per_part = dys.groupby(['speed','language'])['running_time'].mean().reset_index()
# pivot to have both conditions
pivot = per_part.pivot(index='speed', columns='language', values='running_time')
paired = pivot.dropna()  # participants with both conditions

paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit') if len(paired) > 1 else None
# Cohen's dz
if len(paired) > 1:
    diff = paired[1] - paired[0]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
else:
    dz = np.nan

# Summaries
print('Dyslexic sample size rows:', len(dys))
print('Groups: rv0', rv0.count(), 'rv1', rv1.count())
print('Means (running_time): rv0', mean0, 'rv1', mean1)
print('Medians: rv0', rv0.median(), 'rv1', rv1.median())
print('Welch t-test:', welch_t)
print('Mann-Whitney U:', mannwhit)
print('Cohen d (independent):', cohen_d)
print('Paired participants with both conditions:', len(paired))
print('Paired t-test:', paired_t)
print('Cohen dz:', dz)

# Also compute % difference
pct_diff = (mean1 - mean0) / mean0 * 100
print('Percent difference (rv1 vs rv0):', pct_diff)
