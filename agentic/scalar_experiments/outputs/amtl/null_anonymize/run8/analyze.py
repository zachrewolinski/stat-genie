import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.shape)
print(df.isna().sum())
print(df['feature8'].value_counts())
print(df['feature1'].value_counts())
