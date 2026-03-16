import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

below0 = (_df['feature3'] < 0).sum()
above = (_df['feature3'] > _df['feature4']).sum()
print('n rows', len(_df))
print('feature3 < 0:', below0)
print('feature3 > feature4:', above)
print('feature3 min/max', _df['feature3'].min(), _df['feature3'].max())
print('feature4 min/max', _df['feature4'].min(), _df['feature4'].max())
print('feature3/feature4 stats:')
ratio = _df['feature3'] / _df['feature4']
print(ratio.describe())
