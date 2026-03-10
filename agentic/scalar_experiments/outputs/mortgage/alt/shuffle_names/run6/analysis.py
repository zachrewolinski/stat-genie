import pandas as pd
import json

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.describe(include='all').T.head(20))
