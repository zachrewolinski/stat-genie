import pandas as pd
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print(amtl.dtypes)
print('rows', len(amtl))
print('num_amtl min/max', amtl['num_amtl'].min(), amtl['num_amtl'].max())
print('num_amtl sample unique', amtl['num_amtl'].head(10).tolist())
print('sockets min/max', amtl['sockets'].min(), amtl['sockets'].max())
print('sockets unique', sorted(amtl['sockets'].unique())[:10])
print('num_amtl is int-like', np.allclose(amtl['num_amtl'], np.round(amtl['num_amtl'])))
print('num_amtl within 0..sockets proportion', ((amtl['num_amtl']>=0) & (amtl['num_amtl']<=amtl['sockets'])).mean())

print('genus counts', amtl['genus'].value_counts())

# check if there is any column for outcome like amtl proportion
for col in amtl.columns:
    if 'amtl' in col and col!='num_amtl':
        print('amtl col', col)

# compute ratio if possible
amtl['amtl_rate'] = amtl['num_amtl'] / amtl['sockets']
print('amtl_rate min/max', amtl['amtl_rate'].min(), amtl['amtl_rate'].max())

# missing values
print('missing per column', amtl.isna().mean().sort_values(ascending=False).head(10))

