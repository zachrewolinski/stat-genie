import pandas as pd

path = 'affairs.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print('\ninfo')
print(df.info())
print('\nunique counts')
for col in df.columns:
    print(col, df[col].nunique())
print('\nsummary numeric')
print(df.describe())
print('\nsummary all')
print(df.describe(include='all'))
