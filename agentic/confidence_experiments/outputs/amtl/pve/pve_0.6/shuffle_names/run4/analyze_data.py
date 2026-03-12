import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)
print('\nSummary:')
print(df.describe(include='all'))
