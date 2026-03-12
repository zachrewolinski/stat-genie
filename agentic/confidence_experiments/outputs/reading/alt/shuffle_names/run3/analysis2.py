import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

col='running_time'
print(df[col].describe(percentiles=[0.5,0.9,0.95,0.99,0.999]))
print('top5', df[col].sort_values(ascending=False).head().to_list())

num_words = df['num_words']
for time_col in ['adjusted_running_time','age','gender','running_time','retake_trial']:
    t = df[time_col]
    wpm_ms = num_words / (t/60000)
    wpm_s = num_words / (t/60)
    print(time_col, 'median wpm if ms', np.nanmedian(wpm_ms), 'median wpm if s', np.nanmedian(wpm_s))
