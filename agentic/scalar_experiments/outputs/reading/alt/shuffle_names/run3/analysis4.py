import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for col in ['adjusted_running_time','age','gender']:
    corr = df['running_time'].corr(df[col])
    print('corr running_time vs', col, corr)

# compute speed from words/time if retake_trial is word count and adjusted_running_time is ms
word_col = 'retake_trial'
for time_col in ['adjusted_running_time','age','gender']:
    wpm = df[word_col] / (df[time_col] / 60000)
    corr = df['running_time'].corr(wpm)
    print('corr running_time vs wpm from', time_col, corr)
    print(time_col, 'wpm median', wpm.median(), 'mean', wpm.mean())

