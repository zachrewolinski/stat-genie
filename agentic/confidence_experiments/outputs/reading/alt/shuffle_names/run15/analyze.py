import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)

# candidate word count columns
word_cols = ['retake_trial', 'num_words']
# candidate time columns
time_cols = ['adjusted_running_time', 'age', 'gender', 'running_time']

print('correlations with word counts:')
for w in word_cols:
    for t in time_cols:
        if df[w].dtype != 'object' and df[t].dtype != 'object':
            corr = df[w].corr(df[t])
            print(f'{w} vs {t}: {corr:.4f}')

# check ratio of adjusted_running_time to running_time
for t in ['adjusted_running_time','age']:
    ratio = df[t] / df['running_time']
    print(f'ratio {t}/running_time: mean {ratio.mean():.2f} median {ratio.median():.2f}')

# compute WPM using retake_trial and time columns
for t in ['adjusted_running_time','age','running_time']:
    time = df[t].astype(float)
    words = df['retake_trial'].astype(float)
    # assume time is ms for adjusted_running_time/age
    if t in ['adjusted_running_time','age']:
        wpm = words * 60000 / time
    else:
        # assume running_time is seconds
        wpm = words * 60 / time
    print(f'WPM (words from retake_trial) using {t}: mean {wpm.mean():.1f} median {wpm.median():.1f} min {wpm.min():.1f} max {wpm.max():.1f}')

# check correlation between computed WPM and running_time
wpm_adj = df['retake_trial'] * 60000 / df['adjusted_running_time']
print('corr wpm_adj vs running_time', wpm_adj.corr(df['running_time']))

# check correlation between adjusted and age
print('corr adjusted_running_time vs age', df['adjusted_running_time'].corr(df['age']))

# check if running_time equals adjusted_running_time/??
print('running_time summary', df['running_time'].describe())
print('adjusted_running_time summary', df['adjusted_running_time'].describe())

# check group stats: running_time by language (binary)
print('\nrunning_time by language')
print(df.groupby('language')['running_time'].describe()[['mean','median','count','std']])

# check group stats: adjusted_running_time by language
print('\nadjusted_running_time by language')
print(df.groupby('language')['adjusted_running_time'].describe()[['mean','median','count','std']])

# check group stats: wpm_adj by language
print('\nwpm_adj by language')
print(df.assign(wpm_adj=wpm_adj).groupby('language')['wpm_adj'].describe()[['mean','median','count','std']])

# dyslexia distribution
print('\ndyslexia counts', df['dyslexia'].value_counts(dropna=False))
