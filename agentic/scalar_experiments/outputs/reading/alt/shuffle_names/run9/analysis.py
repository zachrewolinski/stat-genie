import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Identify variables
reader_view = 'language'  # 0/1 indicator
# Dyslexia status likely in 'device' (0=no, 1=dyslexia, 2=severe)
# We'll treat >0 as dyslexia

# Basic sanity
print('Total rows', len(df))
print('Reader view value counts')
print(df[reader_view].value_counts(dropna=False))

# Determine dyslexia counts for candidate columns
for col in ['device','dyslexia','dyslexia_bin']:
    if col in df.columns:
        print('\n', col, 'value counts')
        print(df[col].value_counts(dropna=False).sort_index())

# Use device as dyslexia status
work = df[[reader_view,'device','running_time','speed']].copy()
work = work.dropna(subset=[reader_view,'device','running_time'])
work['dyslexia_group'] = np.where(work['device']>0, 1, 0)

# Filter dyslexic participants
wd = work[work['dyslexia_group']==1]
print('\nDyslexia rows', len(wd))
print('Unique participants', wd['speed'].nunique())

# Check if participants have both conditions
pivot_counts = wd.pivot_table(index='speed', columns=reader_view, values='running_time', aggfunc='size')
print('\nParticipants with both conditions', ((pivot_counts[0]>0) & (pivot_counts[1]>0)).sum())
print('Participants with only one condition', ((pivot_counts[0].isna()) | (pivot_counts[1].isna())).sum())

# Compute participant-level means per condition
means = wd.groupby(['speed', reader_view])['running_time'].mean().unstack(reader_view)
paired = means.dropna()
print('Paired participants', len(paired))

# Paired t-test
if len(paired) > 2:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    diff = paired[1] - paired[0]
    print('\nPaired t-test (RV=1 vs 0): t', t_stat, 'p', p_val)
    print('Mean diff', diff.mean(), 'median diff', diff.median())
    # effect size: Cohen d for paired
    d = diff.mean() / diff.std(ddof=1)
    print('Cohen d (paired)', d)

# Also compare using independent test on all rows
rv0 = wd[wd[reader_view]==0]['running_time']
rv1 = wd[wd[reader_view]==1]['running_time']
print('\nRow-level counts: rv0', len(rv0), 'rv1', len(rv1))
if len(rv0)>1 and len(rv1)>1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('Welch t-test: t', t_stat, 'p', p_val)
    print('Means', rv1.mean(), rv0.mean(), 'medians', rv1.median(), rv0.median())

# Robust check: log transform (avoid nonpositive)
wd = wd[wd['running_time']>0]
wd['log_speed'] = np.log(wd['running_time'])
rv0 = wd[wd[reader_view]==0]['log_speed']
rv1 = wd[wd[reader_view]==1]['log_speed']
if len(rv0)>1 and len(rv1)>1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('Welch t-test log-speed: t', t_stat, 'p', p_val)
    print('Means log', rv1.mean(), rv0.mean())

# Summary stats
print('\nSummary stats dyslexia group (rows):')
print(wd.groupby(reader_view)['running_time'].agg(['count','mean','median','std']))

