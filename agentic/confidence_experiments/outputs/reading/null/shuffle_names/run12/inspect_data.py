import pandas as pd

df = pd.read_csv('reading.csv')
print(df.head())
print(df.dtypes)
print(df.nunique())
