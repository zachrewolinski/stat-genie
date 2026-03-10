import pandas as pd

df = pd.read_csv('crofoot.csv')
print(df.head())
print('\nColumns:', list(df.columns))
print('\nDescribe:')
print(df.describe(include='all'))
print('\nUnique counts:')
print(df.nunique())
