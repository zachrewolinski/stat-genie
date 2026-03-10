import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for time_col in ['adjusted_running_time','age']:
    # convert from centiseconds
    wpm_cs = df['num_words'] * 6000 / df[time_col]
    corr = wpm_cs.corr(df['running_time'])
    print('corr running_time vs wpm_cs using', time_col, corr)
    print('wpm_cs stats', time_col, wpm_cs.describe())

