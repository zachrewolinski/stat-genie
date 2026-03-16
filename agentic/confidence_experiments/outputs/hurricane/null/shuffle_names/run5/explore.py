import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('hurricane.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nDescribe:')
print(df.describe(include='all'))
