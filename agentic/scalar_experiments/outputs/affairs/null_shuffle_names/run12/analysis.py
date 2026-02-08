import pandas as pd

_df = pd.read_csv('affairs.csv')
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all'))
print('columns', _df.columns.tolist())
print('unique children', _df['children'].unique()[:10])
