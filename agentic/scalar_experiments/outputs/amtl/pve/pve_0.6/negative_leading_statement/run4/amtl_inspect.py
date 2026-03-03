import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df[['num_amtl','sockets']].describe())
print(df.groupby('genus')['num_amtl'].agg(['mean','std','count']))
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
print(df.groupby('genus')['num_amtl'].min())
print(df.groupby('genus')['num_amtl'].max())
# logit back to p (if num_amtl is logit)
p = 1/(1+np.exp(-df['num_amtl']))
print('logit back to p summary', p.describe())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
