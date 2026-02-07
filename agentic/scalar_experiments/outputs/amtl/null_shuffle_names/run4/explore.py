import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('\ncolumns:', list(df.columns))
print('\ndtypes:\n', df.dtypes)
print('\nmissing:\n', df.isna().sum())
print('\nunique:\n', df.nunique())
