import pandas as pd
import json

df = pd.read_csv('amtl.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nUnique counts:')
print(df.nunique())
print('\nDescribe:')
print(df.describe(include='all'))
