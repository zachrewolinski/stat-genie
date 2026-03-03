import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
_df['missing'] = np.exp(_df['genus'])
_df['total'] = _df['age']
_df['prop'] = _df['missing'] / _df['total']
print('prop min', _df['prop'].min(), 'max', _df['prop'].max())
print('prop >1 count', (_df['prop']>1).sum())
print('prop >1 fraction', (_df['prop']>1).mean())
print('missing>total count', (_df['missing']>_df['total']).sum())
print('missing>total max ratio', (_df['missing']/_df['total']).max())

