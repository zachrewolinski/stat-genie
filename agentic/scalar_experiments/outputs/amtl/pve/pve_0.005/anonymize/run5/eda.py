import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)
print('feature1 unique', df['feature1'].unique()[:10])
print('feature8 unique', df['feature8'].unique())
print('feature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())
print('feature5 min/max', df['feature5'].min(), df['feature5'].max())
print('feature7 unique', sorted(df['feature7'].unique()))
