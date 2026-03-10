import pandas as pd
import numpy as np

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print(df.info())
print(df.describe(include='all'))
print('missing', df.isna().sum())
