import pandas as pd
import json

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
# basic stats
print(df.describe(include='all').transpose().head(20))
