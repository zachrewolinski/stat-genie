import pandas as pd
import json

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.nunique())
print(df.describe(include='all'))
