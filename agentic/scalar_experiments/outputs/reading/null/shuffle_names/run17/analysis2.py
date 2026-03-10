import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')

# compute speed candidates
speed_wpm = df['num_words'] / (df['age'] / 60000.0)  # if age is adjusted_running_time? hmm
speed_wpm2 = df['num_words'] / (df['adjusted_running_time'] / 60000.0)

# compare with running_time
for name, series in [('speed_wpm', speed_wpm), ('speed_wpm2', speed_wpm2)]:
    corr = series.corr(df['running_time'])
    print(name, 'corr with running_time', corr)

# print summary of running_time
print(df['running_time'].describe())

# check if running_time derived from num_words and age
# compute implied time if running_time were speed wpm: time = num_words / (wpm/60000)
# wpm -> time = num_words / wpm * 60000
implied_time_from_running = df['num_words'] / df['running_time'] * 60000
print('implied_time_from_running desc', implied_time_from_running.describe())

# correlations with adjusted_running_time / age
print('corr implied_time_from_running with adjusted_running_time', implied_time_from_running.corr(df['adjusted_running_time']))
print('corr implied_time_from_running with age', implied_time_from_running.corr(df['age']))
