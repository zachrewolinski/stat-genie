import pandas as pd

_df = pd.read_csv('amtl.csv')
_df = _df.dropna(subset=['num_amtl','sockets'])

invalid = _df[(_df['num_amtl'] < 0) | (_df['num_amtl'] > _df['sockets']) | (_df['sockets'] <= 0)]
print('rows', _df.shape[0])
print('invalid', invalid.shape[0])
print(invalid.head())

print('min sockets', _df['sockets'].min(), 'max', _df['sockets'].max())
print('min num_amtl', _df['num_amtl'].min(), 'max', _df['num_amtl'].max())

print(_df[['age','prob_male','tooth_class','genus']].isna().sum())
