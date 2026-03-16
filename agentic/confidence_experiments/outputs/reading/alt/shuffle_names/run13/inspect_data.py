import pandas as pd

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print('head')
print(df.head())
print('dtypes')
print(df.dtypes)
print('describe numeric')
print(df.describe().transpose())
print('describe all (first 10)')
print(df.describe(include='all').transpose().head(10))
