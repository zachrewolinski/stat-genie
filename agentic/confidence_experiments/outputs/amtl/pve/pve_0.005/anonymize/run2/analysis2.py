import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
_df['ratio'] = _df['feature3'] / _df['feature4']
print(_df['ratio'].describe())
print('ratio min/max', _df['ratio'].min(), _df['ratio'].max())
print('feature3 negative count', (_df['feature3']<0).sum())
print('ratio negative count', (_df['ratio']<0).sum())
print('ratio >1 count', (_df['ratio']>1).sum())

print('mean ratio by genus')
print(_df.groupby('feature8')['ratio'].mean())
print('mean feature3 by genus')
print(_df.groupby('feature8')['feature3'].mean())

