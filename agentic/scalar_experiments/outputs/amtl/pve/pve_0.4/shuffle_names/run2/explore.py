import pandas as pd

_df = pd.read_csv('amtl.csv')
print('columns:', _df.columns.tolist())
print(_df.head())
print(_df.dtypes)
print('nunique:', _df.nunique())

# show unique values for categorical-like columns
for col in _df.columns:
    if _df[col].dtype == 'object':
        print('\n', col, 'unique sample:', _df[col].unique()[:10], 'nunique', _df[col].nunique())

print('\nNumeric describe:')
print(_df.describe())
