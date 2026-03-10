import pandas as pd

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)
print('\nDescribe:')
print(df.describe(include='all'))
