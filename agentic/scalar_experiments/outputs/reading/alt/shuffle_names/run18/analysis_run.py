import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', 50)

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Map columns based on info.json descriptions
# reader_view indicator: column 'language' (0/1)
# dyslexia indicator: column 'correct_rate' (0/1)
# reading speed: column 'running_time' (words per minute)

# Filter to dyslexia participants
sub = df[df['correct_rate'] == 1].copy()
sub = sub[sub['language'].notna() & sub['running_time'].notna() & sub['speed'].notna()]

print('Rows (dyslexia):', len(sub))
print('Unique participants (dyslexia):', sub['speed'].nunique())

# Descriptive stats by condition
summary = sub.groupby('language')['running_time'].agg(['count','mean','median','std']).rename(index={0:'no_reader_view',1:'reader_view'})
print('\nSummary by reader_view condition (rows):')
print(summary)

# Two-sample Welch t-test (row-level)
cond0 = sub.loc[sub['language'] == 0, 'running_time']
cond1 = sub.loc[sub['language'] == 1, 'running_time']

t_welch = stats.ttest_ind(cond1, cond0, equal_var=False, nan_policy='omit')
print('\nWelch t-test (row-level):', t_welch)

# Mann-Whitney U test (row-level)
try:
    u_stat, u_p = stats.mannwhitneyu(cond1, cond0, alternative='two-sided')
    print('Mann-Whitney U (row-level):', u_stat, u_p)
except Exception as e:
    print('Mann-Whitney U failed:', e)

# Participant-level paired analysis
# Compute mean speed per participant per condition
pivot = sub.pivot_table(index='speed', columns='language', values='running_time', aggfunc='mean')

paired = pivot.dropna(subset=[0,1])
print('\nParticipants with both conditions:', len(paired))

# Paired t-test
if len(paired) > 1:
    t_paired = stats.ttest_rel(paired[1], paired[0])
    print('Paired t-test (participant means):', t_paired)

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(paired[1], paired[0])
        print('Wilcoxon signed-rank:', w_stat, w_p)
    except Exception as e:
        print('Wilcoxon failed:', e)

    # Effect size (paired Cohen d)
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1)
    print('Paired Cohen d:', d)

# Report mean difference
mean_diff = cond1.mean() - cond0.mean()
print('\nMean difference (reader_view - no_reader_view) row-level:', mean_diff)

# Participant-level mean difference
if len(paired) > 0:
    mean_diff_paired = (paired[1] - paired[0]).mean()
    print('Mean difference (participant means):', mean_diff_paired)
