import pandas as pd

# Load data
_df = pd.read_csv('amtl.csv')

print('Rows', len(_df))
print('Columns', list(_df.columns))
print('\nDtypes:')
print(_df.dtypes)

print('\nUnique counts:')
for c in _df.columns:
    print(c, _df[c].nunique(dropna=False))

print('\nSample unique values:')
for c in _df.columns:
    vals = _df[c].dropna().unique()[:5]
    print(c, vals)

print('\nNumeric summary:')
print(_df.describe(include='number').T)
