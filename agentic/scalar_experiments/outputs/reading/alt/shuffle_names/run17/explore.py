import pandas as pd

df = pd.read_csv('reading.csv')
print('shape', df.shape)
print(df.head())
print(df.dtypes)
print('missing top', df.isna().mean().sort_values(ascending=False).head())
