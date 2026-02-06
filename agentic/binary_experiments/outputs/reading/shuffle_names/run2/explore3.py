import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate time cols
for time_col in ['adjusted_running_time','age','gender']:
    speed = df['num_words'] / (df[time_col] / 60000.0)
    print(time_col, speed.describe())

# compare running_time to speed
for col in ['adjusted_running_time','age','gender']:
    speed = df['num_words'] / (df[col] / 60000.0)
    corr = speed.corr(df['running_time'])
    print('corr speed from', col, 'with running_time', corr)
