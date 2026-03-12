import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all'))

# Check num_amtl distribution
print('num_amtl min/max', _df['num_amtl'].min(), _df['num_amtl'].max())
print('num_amtl unique negative count', (_df['num_amtl']<0).sum())

# Check sockets values
print('sockets min/max', _df['sockets'].min(), _df['sockets'].max())
print('sockets unique', sorted(_df['sockets'].unique())[:10])

# Check genus counts
print(_df['genus'].value_counts())

# Check tooth_class counts
print(_df['tooth_class'].value_counts())

# Check prob_male distribution
print(_df['prob_male'].value_counts())

# Check if num_amtl close to integers? maybe standardized
print('num_amtl mean', _df['num_amtl'].mean(), 'std', _df['num_amtl'].std())
print('sockets mean', _df['sockets'].mean())

# maybe there is raw num_amtl? Check if num_amtl equals (raw - mean)/std? Not sure.
# Let's check if num_amtl has many decimals and if num_amtl*sockets could be integer? compute proportion
prop = _df['num_amtl']/_df['sockets']
print('prop min/max', prop.min(), prop.max())

