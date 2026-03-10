import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
num_words = df['retake_trial'].astype(float)
time_ms = df['age'].astype(float)
calc_speed = num_words / (time_ms/60000.0)
max_diff = np.nanmax(np.abs(calc_speed - df['running_time'].astype(float)))
print('max abs diff', max_diff)
print('corr', np.corrcoef(calc_speed, df['running_time'].astype(float))[0,1])
