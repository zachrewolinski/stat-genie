import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

# numeric columns
num_cols = df.select_dtypes(include=['number']).columns
print('numeric columns', num_cols)

# describe
print('\nSummary stats')
print(df[num_cols].describe().T[['mean','std','min','max']])

# check correlations among numeric columns
corr = df[num_cols].corr()
print('\nTop correlations with num_words')
for col in num_cols:
    if col == 'num_words':
        continue
    print(col, corr.loc['num_words', col])

print('\nTop correlations with retake_trial')
for col in num_cols:
    if col == 'retake_trial':
        continue
    print(col, corr.loc['retake_trial', col])

print('\nTop correlations with running_time')
for col in num_cols:
    if col == 'running_time':
        continue
    print(col, corr.loc['running_time', col])

# compute estimated reading speed from word counts and running times if plausible
# assume adjusted_running_time or age (ms) or running_time (maybe seconds) convert to wpm
candidates = ['adjusted_running_time', 'age', 'gender']
for time_col in candidates:
    # treat time in ms
    wpm = df['retake_trial'] / (df[time_col] / 1000.0) * 60
    print('\nWPM using retake_trial words and', time_col, 'ms: min', wpm.min(), 'max', wpm.max(), 'mean', wpm.mean())

# check if num_words could be wpm
print('\nnum_words as wpm? min', df['num_words'].min(), 'max', df['num_words'].max(), 'mean', df['num_words'].mean())
