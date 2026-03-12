import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

word_cols=['num_words','retake_trial']
time_cols=['adjusted_running_time','age','gender']

for word_col in word_cols:
    for time_col in time_cols:
        # assume time in ms
        wpm = df[word_col] / (df[time_col] / 60000.0)
        desc = wpm.describe(percentiles=[0.01,0.05,0.5,0.95,0.99])
        print(f"\nWPM using {word_col} and {time_col}")
        print(desc)
