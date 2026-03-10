import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

col='running_time'
print(df[col].describe(percentiles=[0.5,0.9,0.95,0.99,0.999]))

# check top 5 values
print(df[col].sort_values(ascending=False).head())

# check relation: maybe running_time corresponds to speed? compute words/time
num_words = df['num_words']
# try compute wpm using adjusted_running_time (ms)
for time_col in ['adjusted_running_time','age','gender','running_time','retake_trial']:
    t = df[time_col]
    # if time in ms -> convert to minutes
    wpm_ms = num_words / (t/60000)
    wpm_s = num_words / (t/60)
    # show medians
    print(time_col, 'wpm_ms median', np.nanmedian(wpm_ms), 'wpm_s median', np.nanmedian(wpm_s))

