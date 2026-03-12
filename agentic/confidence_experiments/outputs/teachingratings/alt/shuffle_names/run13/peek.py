import pandas as pd

path = 'teachingratings.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('na counts')
print(df.isna().sum())
