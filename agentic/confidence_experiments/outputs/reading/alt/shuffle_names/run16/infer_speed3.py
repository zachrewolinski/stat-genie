import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

adj_time = df['age']
num_words = df['num_words']

wpm_seconds = num_words * 60 / adj_time  # if adj_time in seconds

print(wpm_seconds.describe())
print('corr with running_time', np.corrcoef(wpm_seconds, df['running_time'])[0,1])
print('median abs diff', (df['running_time'] - wpm_seconds).abs().median())

# maybe wpm_seconds * 1000? (if adj_time in ms)
for scale in [1, 1000, 0.001, 0.01, 0.1]:
    diff = (df['running_time'] - wpm_seconds*scale).abs().median()
    print('scale', scale, 'median diff', diff)
