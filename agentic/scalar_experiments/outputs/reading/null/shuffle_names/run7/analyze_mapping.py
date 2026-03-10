import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')

# candidate speed computations
for time_col in ['adjusted_running_time','age','gender']:
    # avoid zero
    wpm = df['num_words'] / df[time_col] * 60000
    corr = wpm.corr(df['running_time'])
    print('corr running_time vs wpm using', time_col, corr)
    print('wpm stats', time_col, wpm.describe())

# correlation between running_time and times
for col in ['adjusted_running_time','age','gender']:
    print('corr running_time vs', col, df['running_time'].corr(df[col]))

# check if running_time might be time not speed: compare to adjusted_running_time (ms) maybe similar scale?
print('running_time stats', df['running_time'].describe())

# crosstab between device and dyslexia_bin
print('device unique', df['device'].dropna().unique())
print('dyslexia unique', df['dyslexia'].dropna().unique())
print('dyslexia_bin unique', df['dyslexia_bin'].dropna().unique())
print('correct_rate unique', df['correct_rate'].dropna().unique())

print('crosstab device vs dyslexia_bin')
print(pd.crosstab(df['device'], df['dyslexia_bin']))

print('crosstab dyslexia vs dyslexia_bin')
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin']))

print('crosstab device vs correct_rate')
print(pd.crosstab(df['device'], df['correct_rate']))

print('crosstab dyslexia vs correct_rate')
print(pd.crosstab(df['dyslexia'], df['correct_rate']))

# check if reader_view (language col) correlates with some known indicator? look at balance
print('reader_view (language) value counts')
print(df['language'].value_counts())

