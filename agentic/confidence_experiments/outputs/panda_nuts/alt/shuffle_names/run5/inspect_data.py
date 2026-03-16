import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nShape:', df.shape)
print('\nDescribe:')
print(df.describe(include='all'))
