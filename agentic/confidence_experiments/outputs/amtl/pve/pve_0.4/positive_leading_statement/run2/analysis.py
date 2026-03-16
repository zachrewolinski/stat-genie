import pandas as pd
import numpy as np
from scipy.special import expit


df = pd.read_csv('amtl.csv')
print(df.head())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl mean', df['num_amtl'].mean(), 'std', df['num_amtl'].std())
print('sockets range', df['sockets'].min(), df['sockets'].max())

# check if expit(num_amtl)*sockets near integer
p = expit(df['num_amtl'])
counts = p * df['sockets']
# compute how close to integer
frac = np.abs(counts - np.round(counts))
print('mean frac to nearest int', frac.mean(), 'median', np.median(frac))
print('percent within 0.05 of integer', (frac < 0.05).mean())

# check if num_amtl/sockets within 0-1
prop = df['num_amtl'] / df['sockets']
print('num_amtl/sockets range', prop.min(), prop.max())

print('unique genus', df['genus'].unique())
print('unique tooth_class', df['tooth_class'].unique())

# summary by genus
print(df.groupby('genus')['num_amtl'].agg(['mean','std','count']))
