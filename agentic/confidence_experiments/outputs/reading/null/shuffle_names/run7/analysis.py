import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Map key variables based on inspection
reader_view = df['language']  # 0/1
# Dyslexia status likely in 'device' (0 no, 1 dyslexia, 2 severe)
# Use dyslexic flag for device > 0

# reading speed likely in 'running_time'

df = df.copy()

df['reader_view'] = reader_view

df['dyslexia_status'] = df['device']

df['dyslexic'] = df['dyslexia_status'].isin([1.0, 2.0])

df['reading_speed'] = df['running_time']

# drop missing essentials
analysis_df = df.dropna(subset=['reader_view','dyslexic','reading_speed','speed'])

# Filter to dyslexic participants
analysis_df = analysis_df[analysis_df['dyslexic']].copy()

print('Rows dyslexic:', len(analysis_df))
print('Participants dyslexic:', analysis_df['speed'].nunique())

# group counts by reader_view
print('Counts by reader_view (rows):')
print(analysis_df['reader_view'].value_counts())

# Summary stats (row-level)
summary = analysis_df.groupby('reader_view')['reading_speed'].agg(['count','mean','median','std'])
print('\nRow-level summary:')
print(summary)

# Row-level tests
rv1 = analysis_df.loc[analysis_df['reader_view']==1, 'reading_speed']
rv0 = analysis_df.loc[analysis_df['reader_view']==0, 'reading_speed']

# t-test with unequal variances
print('\nRow-level Welch t-test:')
print(stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit'))

# Mann-Whitney U
print('\nRow-level Mann-Whitney U:')
print(stats.mannwhitneyu(rv1, rv0, alternative='two-sided'))

# Effect size Cohen's d (row-level)
mean_diff = rv1.mean() - rv0.mean()
pooled_sd = np.sqrt(((rv1.var(ddof=1) + rv0.var(ddof=1)) / 2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
print('\nRow-level mean diff', mean_diff, 'Cohen d', cohen_d)

# Participant-level aggregation (mean speed per participant per condition)
# Compute per participant per reader_view mean
participant_means = analysis_df.groupby(['speed','reader_view'])['reading_speed'].mean().reset_index()

# Keep participants with both conditions
pivot = participant_means.pivot(index='speed', columns='reader_view', values='reading_speed')
paired = pivot.dropna()
print('\nParticipants with both conditions:', paired.shape[0])

if paired.shape[0] > 1:
    # Paired t-test
    ttest = stats.ttest_rel(paired[1], paired[0])
    print('Paired t-test:', ttest)
    # Wilcoxon signed-rank
    try:
        wilcoxon = stats.wilcoxon(paired[1] - paired[0])
    except Exception as e:
        wilcoxon = str(e)
    print('Wilcoxon:', wilcoxon)
    # Effect size: Cohen d for paired (mean diff / sd diff)
    diff = paired[1] - paired[0]
    d_paired = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    print('Paired mean diff', diff.mean(), 'Cohen d paired', d_paired)

# Regression with cluster-robust SE by participant
analysis_df['rv'] = analysis_df['reader_view']
model = smf.ols('reading_speed ~ rv', data=analysis_df).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speed']})
print('\nOLS with cluster-robust SE:')
print(model.summary().tables[1])

# Log-transform speed to reduce skew; add small constant to avoid log(0)
analysis_df['log_speed'] = np.log(analysis_df['reading_speed'] + 1e-6)
model_log = smf.ols('log_speed ~ rv', data=analysis_df).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speed']})
print('\nOLS log-speed with cluster-robust SE:')
print(model_log.summary().tables[1])

