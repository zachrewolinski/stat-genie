import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')
# derive wpm from retake_trial (likely num_words) and adjusted_running_time (ms)
# But we also test with age/gender if they are times.

for time_col in ['adjusted_running_time','age','gender']:
    # avoid zero
    wpm = df['retake_trial'] / (df[time_col] / 1000.0 / 60.0)
    corr = wpm.corr(df['running_time'])
    print('corr wpm from retake_trial &', time_col, 'with running_time:', corr)

# try using num_words as words
for time_col in ['adjusted_running_time','age','gender']:
    wpm2 = df['num_words'] / (df[time_col] / 1000.0 / 60.0)
    corr2 = wpm2.corr(df['running_time'])
    print('corr wpm from num_words &', time_col, 'with running_time:', corr2)

# correlation between running_time and adjusted_running_time/time columns
for col in ['adjusted_running_time','age','gender']:
    print('corr running_time vs', col, df['running_time'].corr(df[col]))

# correlation between running_time and retake_trial/num_words
print('corr running_time vs retake_trial', df['running_time'].corr(df['retake_trial']))
print('corr running_time vs num_words', df['running_time'].corr(df['num_words']))

# show quantiles for running_time
print('running_time quantiles', df['running_time'].quantile([0.1,0.25,0.5,0.75,0.9,0.99]))
