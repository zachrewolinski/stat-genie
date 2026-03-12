import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df.head())
print('\nColumns:', _df.columns.tolist())
print('\nDtypes:', _df.dtypes)
print('\nUnique counts:')
print(_df.nunique())

for col in _df.columns:
    if _df[col].dtype == 'object':
        print(f"\n{col} sample unique:", _df[col].dropna().unique()[:10])
    else:
        print(f"\n{col} min/max:", _df[col].min(), _df[col].max())

print('\nMissing values:')
print(_df.isna().mean())
