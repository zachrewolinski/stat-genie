import pandas as pd

df = pd.read_csv('mortgage.csv')
print(df.head())
print(df.columns.tolist())
print(df.shape)
print(df[['female','deny','accept']].describe(include='all'))
print(df[['female','deny','accept']].isna().sum())
