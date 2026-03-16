import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Map columns based on value patterns
participant_id = df['speed']  # UUID
reader_view = df['language']  # 0/1
# Dyslexia severity in 'device' (0 no, 1 dys, 2 severe)
dyslexia_sev = df['device']
# Reading speed (words per minute)
reading_speed = df['running_time']

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'participant': participant_id,
    'reader_view': reader_view,
    'dyslexia_sev': dyslexia_sev,
    'speed_wpm': reading_speed
})

# Keep only rows with needed values
analysis_df = analysis_df.dropna(subset=['reader_view','dyslexia_sev','speed_wpm','participant'])

# Dyslexia binary
analysis_df['dyslexia_bin'] = (analysis_df['dyslexia_sev'] > 0).astype(int)

# Subset to dyslexia participants
dys_df = analysis_df[analysis_df['dyslexia_bin'] == 1]

print('Dyslexia rows', dys_df.shape)
print('Participants with dyslexia', dys_df['participant'].nunique())

# Simple group comparison (all rows)
rv1 = dys_df[dys_df['reader_view'] == 1]['speed_wpm']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed_wpm']

print('\nAll rows comparison:')
print('n rv1', rv1.size, 'n rv0', rv0.size)
print('mean rv1', rv1.mean(), 'mean rv0', rv0.mean())
print('median rv1', rv1.median(), 'median rv0', rv0.median())

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False)
print('Welch t-test', welch)

# Effect size (Cohen d)
# compute pooled SD for independent samples using standard formula with unequal n
n1, n0 = rv1.size, rv0.size
s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2))
cohen_d = (rv1.mean() - rv0.mean())/sp
print('Cohen d', cohen_d)

# Participant-level paired comparison
# compute per participant mean speed under each reader_view
pt_means = dys_df.groupby(['participant','reader_view'])['speed_wpm'].mean().unstack()
# keep participants with both conditions
paired = pt_means.dropna()
print('\nPaired participants', paired.shape[0])
if paired.shape[0] > 1:
    diff = paired[1] - paired[0]
    # paired t-test
    paired_t = stats.ttest_rel(paired[1], paired[0])
    print('Paired t-test', paired_t)
    print('mean diff', diff.mean(), 'median diff', diff.median())
    print('diff 5-95', np.percentile(diff, [5,95]))
    # effect size for paired (Cohen dz)
    dz = diff.mean() / diff.std(ddof=1)
    print('Cohen dz', dz)

# Optional: nonparametric test
if rv1.size > 0 and rv0.size > 0:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('\nMann-Whitney U', mw)
