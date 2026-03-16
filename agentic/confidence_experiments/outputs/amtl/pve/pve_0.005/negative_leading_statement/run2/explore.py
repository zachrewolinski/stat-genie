import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# basic group stats
stats = df.groupby('genus')['num_amtl'].agg(['count','mean','std','min','max'])
print(stats)

# check correlation with sockets
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
print('corr num_amtl vs age', df['num_amtl'].corr(df['age']))

# check if num_amtl roughly proportional to sockets by genus
for g, gdf in df.groupby('genus'):
    print('\n', g)
    print('mean sockets', gdf['sockets'].mean())
    print('mean num_amtl', gdf['num_amtl'].mean())
    print('corr num_amtl vs sockets', gdf['num_amtl'].corr(gdf['sockets']))

# check if num_amtl within plausible range of sockets (e.g., less than sockets)
print('num_amtl > sockets count', (df['num_amtl']>df['sockets']).sum())
print('num_amtl < 0 count', (df['num_amtl']<0).sum())

# check per tooth_class
print(df.groupby('tooth_class')['num_amtl'].agg(['mean','std','min','max']))
