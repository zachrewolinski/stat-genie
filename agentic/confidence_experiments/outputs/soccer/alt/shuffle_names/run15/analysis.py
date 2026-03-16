import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all').transpose().head(30))

# check missing
print('missing pct')
print(df.isna().mean().sort_values(ascending=False).head(20))

