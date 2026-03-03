import pandas as pd

path='amtl.csv'

df=pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('feature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())
print('feature3 unique sample', df['feature3'].head(10).tolist())
print('feature4 unique sample', df['feature4'].head(10).tolist())
print('feature3 non-integer count', ((df['feature3']%1)!=0).sum())
print('feature4 non-integer count', ((df['feature4']%1)!=0).sum())
print('feature3 negative count', (df['feature3']<0).sum())
