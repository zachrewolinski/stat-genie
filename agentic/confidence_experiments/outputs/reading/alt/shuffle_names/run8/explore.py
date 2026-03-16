import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

wpm = df['num_words'] / (df['adjusted_running_time']/60000)
print('wpm summary')
print(wpm.describe())

for col in ['adjusted_running_time','age','gender','running_time']:
    print('\n',col)
    print(df[col].describe())

for col in ['adjusted_running_time','age','gender','running_time']:
    corr = np.corrcoef(wpm, df[col])[0,1]
    print('corr wpm vs', col, corr)

for col in ['adjusted_running_time','age','gender','running_time']:
    corr = np.corrcoef(df['num_words'], df[col])[0,1]
    print('corr num_words vs', col, corr)

print('running_time min/max', df['running_time'].min(), df['running_time'].max())
print('wpm min/max', wpm.min(), wpm.max())

ratio = df['running_time'] / wpm
print('ratio summary')
print(ratio.describe())
