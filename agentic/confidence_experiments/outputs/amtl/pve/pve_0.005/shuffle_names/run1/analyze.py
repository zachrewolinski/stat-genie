import pandas as pd
import json

# Load
_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print('nrows', len(_df))
# unique counts
for col in _df.columns:
    print('\n', col)
    print(_df[col].head())
    print('nunique', _df[col].nunique())
    if _df[col].dtype == 'object':
        print('unique sample', _df[col].dropna().unique()[:10])
    else:
        print(_df[col].describe())
