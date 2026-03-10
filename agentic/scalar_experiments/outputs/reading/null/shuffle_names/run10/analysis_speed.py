import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

# correlations of running_time and adjusted_running_time with num_words
for col in ['running_time','adjusted_running_time','age','gender']:
    if col in df.columns:
        corr=df[col].corr(df['num_words'])
        print(col,'corr with num_words',corr)

# compute words per minute from adjusted_running_time and running_time (assuming ms for adjusted, seconds for running)
# If adjusted_running_time is ms
wpm_adj = df['num_words'] / (df['adjusted_running_time']/1000) * 60
print('wpm_adj stats', wpm_adj.describe())
# If running_time is seconds
wpm_run = df['num_words'] / df['running_time'] * 60
print('wpm_run stats', wpm_run.describe())

# check for unrealistic values
print('wpm_adj > 1000', (wpm_adj>1000).mean())
print('wpm_run > 1000', (wpm_run>1000).mean())
