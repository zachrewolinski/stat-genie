import pandas as pd

pd.set_option('display.max_columns', None)


df = pd.read_csv('amtl.csv')
print('columns:', df.columns.tolist())
print(df.head())
print(df.dtypes)
print('nrows', len(df))
print(df.isna().sum())
print(df.describe(include='all'))
