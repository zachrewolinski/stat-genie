import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
print(_df.groupby('sockets')['age'].describe())
print(_df.groupby('sockets')['num_amtl'].describe())
print(_df.groupby('sockets')['genus'].describe())

# Check if age <= num_amtl or such
print('age<=num_amtl fraction', np.mean(_df['age'] <= _df['num_amtl']))
print('age<=num_amtl min diff', (_df['num_amtl'] - _df['age']).min(), (_df['num_amtl'] - _df['age']).max())

# check correlation between age and num_amtl/genus/pop
print(_df[['age','num_amtl','genus','pop','stdev_age']].corr())

