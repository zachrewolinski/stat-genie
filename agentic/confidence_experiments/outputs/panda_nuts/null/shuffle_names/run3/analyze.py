import pandas as pd

path = 'panda_nuts.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('unique nuts_opened', df['nuts_opened'].unique())
print('unique sex', df['sex'].unique())
print('unique seconds', df['seconds'].unique())
