import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('min feature3', df['feature3'].min(), 'max', df['feature3'].max())
print('unique feature1', df['feature1'].unique()[:10])
print('unique feature8', df['feature8'].unique())
print('feature4 min', df['feature4'].min())
print('feature3 negative count', (df['feature3']<0).sum())
print('feature3 integer fraction', ((df['feature3'] % 1)==0).mean())
print('feature4 integer fraction', ((df['feature4'] % 1)==0).mean())

# check if feature3 maybe already standardized per sockets? compute ratio
if (df['feature4']>0).all():
    df['rate'] = df['feature3']/df['feature4']
    print('rate min', df['rate'].min(), 'max', df['rate'].max())
    print(df['rate'].describe())

