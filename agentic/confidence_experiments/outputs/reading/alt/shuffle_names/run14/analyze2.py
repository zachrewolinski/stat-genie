import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# compute correlations between numeric columns
num = df.select_dtypes(include=[np.number])
print('numeric cols', num.columns.tolist())
print(num.describe().T[['mean','std','min','max']])

# pairwise correlations with running_time
for col in num.columns:
    if col == 'running_time':
        continue
    corr = num['running_time'].corr(num[col])
    print('corr running_time', col, corr)

# check if running_time equals num_words / adjusted_running_time * 60000 etc
# We'll compute derived speeds
if 'num_words' in df.columns:
    df['speed_from_adjusted_ms'] = df['num_words'] * 60000 / df['adjusted_running_time']
    df['speed_from_running_ms'] = df['num_words'] * 60000 / df['age']
    df['speed_from_adjusted_s'] = df['num_words'] * 60 / df['adjusted_running_time']
    df['speed_from_running_s'] = df['num_words'] * 60 / df['age']
    print('compare running_time to derived (ms) corr:', df['running_time'].corr(df['speed_from_adjusted_ms']))
    print('compare running_time to derived (age ms) corr:', df['running_time'].corr(df['speed_from_running_ms']))
    print('compare running_time to derived (adjusted s) corr:', df['running_time'].corr(df['speed_from_adjusted_s']))
    print('compare running_time to derived (age s) corr:', df['running_time'].corr(df['speed_from_running_s']))
    print(df[['running_time','speed_from_adjusted_ms','speed_from_running_ms','speed_from_adjusted_s','speed_from_running_s']].head())
