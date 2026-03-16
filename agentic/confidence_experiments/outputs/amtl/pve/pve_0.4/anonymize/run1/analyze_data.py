import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all'))
print('feature3 unique sample:', _df['feature3'].head(10).tolist())
print('feature3 min/max:', _df['feature3'].min(), _df['feature3'].max())
print('feature4 min/max:', _df['feature4'].min(), _df['feature4'].max())
print('feature3 integer check:', (_df['feature3'].dropna() % 1 == 0).all())
print('feature4 integer check:', (_df['feature4'].dropna() % 1 == 0).all())

if (_df['feature4']>0).all():
    prop = _df['feature3'] / _df['feature4']
    print('prop min/max', prop.min(), prop.max())
