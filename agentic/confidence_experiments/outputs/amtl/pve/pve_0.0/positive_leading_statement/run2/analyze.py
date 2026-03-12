import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df['num_amtl'].describe())
print('num_amtl integer-like?', np.allclose(df['num_amtl'], np.round(df['num_amtl'])))
print('unique sample', df['num_amtl'].head(20).tolist())
print('sockets describe', df['sockets'].describe())
print('min num_amtl', df['num_amtl'].min(), 'max', df['num_amtl'].max())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl <0 count', (df['num_amtl']<0).sum())
print('num_amtl > sockets count', (df['num_amtl']>df['sockets']).sum())

print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
print('prob_male unique', sorted(df['prob_male'].unique()))

# maybe num_amtl is standardized; check correlation with sockets
print('corr num_amtl sockets', df['num_amtl'].corr(df['sockets']))

# compute proportion if possible assuming num_amtl is count
if (df['num_amtl']>=0).all():
    df['prop'] = df['num_amtl']/df['sockets']
    print('prop describe', df['prop'].describe())

