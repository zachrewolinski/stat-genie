import pandas as pd

path = 'crofoot.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print('\nNunique')
print(df.nunique())
print('\nDescribe')
print(df.describe())
