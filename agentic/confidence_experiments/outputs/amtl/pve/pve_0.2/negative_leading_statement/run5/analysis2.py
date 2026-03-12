import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

print('num_amtl <0 count', (df['num_amtl']<0).sum())
print('num_amtl > sockets count', (df['num_amtl']>df['sockets']).sum())
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets range', df['sockets'].min(), df['sockets'].max())

# Check if num_amtl could be proportion or logit of proportion
# If proportion in [0,1], check range
print('proportion-like count (0<=num_amtl<=1)', ((df['num_amtl']>=0)&(df['num_amtl']<=1)).mean())

# Check distribution by tooth_class
print(df.groupby('tooth_class')['num_amtl'].describe())
