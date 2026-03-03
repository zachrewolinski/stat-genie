import pandas as pd
import numpy as np

# load data

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df.isna().mean())
print('shape', df.shape)
print(df['feature1'].value_counts(dropna=False).head())
print(df['feature8'].value_counts(dropna=False))
