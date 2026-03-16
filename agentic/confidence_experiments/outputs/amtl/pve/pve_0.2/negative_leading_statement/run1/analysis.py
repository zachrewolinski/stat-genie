import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')
print('shape', _df.shape)
print(_df.head())
print(_df.dtypes)
print('num_amtl min/max', _df['num_amtl'].min(), _df['num_amtl'].max())
print('num_amtl unique sample', _df['num_amtl'].head(10).tolist())
print('sockets unique', sorted(_df['sockets'].unique())[:10])
print('num_amtl integer-like', np.all(np.isclose(_df['num_amtl'], np.round(_df['num_amtl']))))
print('num_amtl integer-like proportion', np.mean(np.isclose(_df['num_amtl'], np.round(_df['num_amtl']))))
print('genus counts', _df['genus'].value_counts())
print('tooth_class counts', _df['tooth_class'].value_counts())
print('prob_male unique', sorted(_df['prob_male'].unique()))
