import pandas as pd


df = pd.read_csv('mortgage.csv')
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.shape)
