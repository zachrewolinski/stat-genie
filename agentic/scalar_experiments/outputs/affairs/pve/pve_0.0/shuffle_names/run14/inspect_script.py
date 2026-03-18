import pandas as pd

path = 'affairs.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print(df.describe(include='all'))
for col in df.columns:
    print('\n', col)
    print(df[col].head(10))
    print(df[col].nunique())
