import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# compute derived speeds
speed_adj_ms = df['num_words'] * 60000 / df['adjusted_running_time']
# compare ratio running_time / speed_adj_ms
ratio = df['running_time'] / speed_adj_ms
print('ratio stats', ratio.describe())

# see first few
print(df[['running_time']].head())
print(speed_adj_ms.head())
