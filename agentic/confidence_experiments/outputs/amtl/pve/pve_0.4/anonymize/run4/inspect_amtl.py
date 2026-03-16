import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print('min feature3', df['feature3'].min(), 'max', df['feature3'].max())
print('min feature4', df['feature4'].min(), 'max', df['feature4'].max())
print('feature3 unique small', df['feature3'].nunique())
print('feature4 unique', df['feature4'].nunique())
print('feature8 unique', df['feature8'].unique())
print('feature1 unique', df['feature1'].unique())
print('feature7 unique', sorted(df['feature7'].unique()))
print('feature3 negative rows count', (df['feature3']<0).sum())
print(df[df['feature3']<0].head())
