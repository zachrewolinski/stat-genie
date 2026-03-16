import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print('num rows', len(_df))
print('num_amtl min/max', _df['num_amtl'].min(), _df['num_amtl'].max())
print('sockets min/max', _df['sockets'].min(), _df['sockets'].max())
print('num_amtl integer?', np.allclose(_df['num_amtl'], np.round(_df['num_amtl'])))
print('sockets integer?', np.allclose(_df['sockets'], np.round(_df['sockets'])))
print('num_amtl negative?', (_df['num_amtl']<0).sum())
print('unique genus', _df['genus'].unique())
print('tooth_class unique', _df['tooth_class'].unique())
