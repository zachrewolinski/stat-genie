import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate time columns
for col in ['running_time','adjusted_running_time','age','gender']:
    if col in df.columns:
        vals = df[col].dropna()
        print(col, 'min', vals.min(), 'max', vals.max(), 'mean', vals.mean())

# compute wpm with running_time and adjusted_running_time, assuming time in seconds
for time_col in ['running_time','adjusted_running_time','age']:
    if time_col in df.columns:
        wpm = df['num_words'] / (df[time_col] / 60)
        print('\nWPM using', time_col, 'summary')
        print(wpm.describe())

# compute wpm if time is milliseconds
for time_col in ['running_time','adjusted_running_time','age']:
    if time_col in df.columns:
        wpm = df['num_words'] / (df[time_col] / 60000)
        print('\nWPM (ms) using', time_col, 'summary')
        print(wpm.describe())
