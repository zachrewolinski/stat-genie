import pandas as pd

file_path = 'reading.csv'

df = pd.read_csv(file_path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all').transpose().head(10))
print('columns', df.columns)
print('shape', df.shape)
