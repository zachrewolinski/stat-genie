import pandas as pd

path = 'panda_nuts.csv'
print('reading', path)
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
