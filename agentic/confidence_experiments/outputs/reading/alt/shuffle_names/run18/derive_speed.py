import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

df = pd.read_csv('reading.csv')

# derived wpm using retake_trial as word count and age as adjusted reading time (ms)
with np.errstate(divide='ignore', invalid='ignore'):
    wpm = df['retake_trial'] * 60000 / df['age']

print('Derived wpm stats:', wpm.describe())
print('running_time stats:', df['running_time'].describe())
print('Correlation derived wpm vs running_time:', wpm.corr(df['running_time']))

# also using adjusted_running_time
wpm_adj = df['retake_trial'] * 60000 / df['adjusted_running_time']
print('Correlation wpm_adj vs running_time:', wpm_adj.corr(df['running_time']))

# check ratio between running_time and wpm
ratio = df['running_time'] / wpm
print('Ratio running_time / derived wpm stats:', ratio.describe())
