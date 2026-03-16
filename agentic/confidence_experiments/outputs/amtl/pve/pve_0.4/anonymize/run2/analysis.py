import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('unique feature1', df['feature1'].unique())
print('unique feature8', df['feature8'].unique())
