import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('feature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())
print('feature8 unique', df['feature8'].unique())
print('feature1 unique', df['feature1'].unique())
print('feature7 unique', sorted(df['feature7'].unique()))
