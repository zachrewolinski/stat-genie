import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate num_words columns
num_candidates = ['retake_trial', 'num_words']
# time candidates (ms)
time_candidates = ['adjusted_running_time', 'age', 'gender']

for ncol in num_candidates:
    for tcol in time_candidates:
        speed = df[ncol] * 60000 / df[tcol]
        corr = speed.corr(df['running_time'])
        print(f'speed from {ncol}/{tcol} corr with running_time: {corr:.3f}')
        # summary
        print('  speed mean', speed.mean(), 'median', speed.median(), 'min', speed.min(), 'max', speed.max())

