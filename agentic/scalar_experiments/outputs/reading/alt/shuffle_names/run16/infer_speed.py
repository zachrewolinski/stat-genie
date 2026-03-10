import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# compute reading speed from num_words and adjusted_running_time
# assume adjusted_running_time in ms -> wpm = words / (ms/60000) = words * 60000 / ms
calc_wpm = df['num_words'] * 60000 / df['adjusted_running_time']
calc_wps = df['num_words'] / (df['adjusted_running_time'] / 1000) # words per second
calc_log = np.log(calc_wpm)

candidates = ['running_time','uuid','age','gender','education']

print('calc_wpm summary', calc_wpm.describe())
print('calc_wps summary', calc_wps.describe())

for col in candidates:
    if df[col].dtype.kind in 'if':
        corr = np.corrcoef(calc_wpm, df[col])[0,1]
        corr_wps = np.corrcoef(calc_wps, df[col])[0,1]
        corr_log = np.corrcoef(calc_log, df[col])[0,1]
        print(col, 'corr_wpm', corr, 'corr_wps', corr_wps, 'corr_log', corr_log)

# check if any column equals calc_wpm within tolerance
for col in candidates:
    if df[col].dtype.kind in 'if':
        diff = np.abs(df[col] - calc_wpm)
        print(col, 'median abs diff to wpm', np.median(diff), 'min diff', diff.min())

