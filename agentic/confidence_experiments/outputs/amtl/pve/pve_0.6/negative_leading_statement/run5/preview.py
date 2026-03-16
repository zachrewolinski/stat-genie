import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('num rows', len(df))
print('unique genus', df['genus'].unique())
