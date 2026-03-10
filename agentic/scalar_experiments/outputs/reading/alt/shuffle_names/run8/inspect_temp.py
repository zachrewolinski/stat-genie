import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

# compute candidate speed
wpm = df['num_words'] / (df['adjusted_running_time']/60000)
# Some columns of time
print('wpm summary', wpm.describe())

for col in ['adjusted_running_time','age','gender','running_time']:
    print('\n',col, df[col].describe())

# correlation between candidate wpm and running_time or age etc
for col in ['adjusted_running_time','age','gender','running_time']:
    corr = np.corrcoef(wpm, df[col])[0,1]
    print('corr wpm vs', col, corr)

# correlation between num_words and times
for col in ['adjusted_running_time','age','gender','running_time']:
    corr = np.corrcoef(df['num_words'], df[col])[0,1]
    print('corr num_words vs', col, corr)

# check if running_time equals wpm or similar
print('running_time min/max', df['running_time'].min(), df['running_time'].max())
print('wpm min/max', wpm.min(), wpm.max())

# compare running_time to wpm with ratio
ratio = df['running_time'] / wpm
print('ratio summary', ratio.describe())
