import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# check missing vs sockets
print('feature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())

# how many feature3 <0 or >feature4
print('feature3<0', (df['feature3']<0).mean())
print('feature3>feature4', (df['feature3']>df['feature4']).mean())

# if feature3 is proportion, check range
print('feature3 between 0 and 1', ((df['feature3']>=0)&(df['feature3']<=1)).mean())

# check if feature3*feature4 close to integer
prod = df['feature3']*df['feature4']
frac = np.abs(prod - np.round(prod))
print('mean abs frac of feature3*feature4 from integer', frac.mean())
print('pct within 0.1 of integer', (frac<0.1).mean())

# correlations
print('corr(feature3, feature4)', df['feature3'].corr(df['feature4']))
print('corr(feature3, age)', df['feature3'].corr(df['feature5']))

# group means by genus
print(df.groupby('feature8')['feature3'].mean())
