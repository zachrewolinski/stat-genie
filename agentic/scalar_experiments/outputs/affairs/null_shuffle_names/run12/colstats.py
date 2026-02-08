import pandas as pd

_df = pd.read_csv('affairs.csv')
for col in _df.columns:
    uniques = sorted(_df[col].dropna().unique())
    print(col, 'dtype', _df[col].dtype, 'nunique', _df[col].nunique())
    print('  min', _df[col].min(), 'max', _df[col].max())
    print('  uniques sample', uniques[:10])
