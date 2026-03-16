import pandas as pd
import numpy as np


df=pd.read_csv('reading.csv')

# time-like columns: adjusted_running_time, age, gender, running_time
cols=['adjusted_running_time','age','gender','running_time']
print(df[cols].describe())

# Check if any columns are close to each other
for a in cols:
    for b in cols:
        if a>=b: continue
        diff = (df[a]-df[b]).abs().median()
        print(a,b,'median abs diff',diff)

# Check if any column is near another times in ms vs seconds
# running_time might be in seconds, so compare adjusted_running_time/1000 or /60
for a in ['adjusted_running_time','age','gender']:
    med_ratio = (df[a]/df['running_time']).median()
    print('median ratio',a,'/ running_time',med_ratio)
    med_ratio2 = (df[a]/(df['running_time']*1000)).median()
    print('median ratio',a,'/ (running_time*1000)',med_ratio2)

# check if adjusted_running_time ~ age - gender or similar
combos=[('age','gender'),('adjusted_running_time','gender'),('adjusted_running_time','age')]
for x,y in combos:
    diff = df[x]-df[y]
    print(f'{x}-{y} stats', diff.min(), diff.max(), diff.mean(), diff.median())

# correlation between time-like columns
print('\nCorrelation:')
print(df[cols].corr())
