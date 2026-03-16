import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for word_col in ['retake_trial','num_words']:
    for time_col in ['adjusted_running_time','age']:
        wpm = df[word_col] / (df[time_col] / 60000.0)
        corr = wpm.corr(df['running_time'])
        print(word_col, time_col, 'corr', corr, 'mean', wpm.mean())
