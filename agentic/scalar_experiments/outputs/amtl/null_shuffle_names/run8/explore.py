import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
print('\nunique genus', df['genus'].unique()[:10])
print('unique tooth_class', df['tooth_class'].unique()[:10])
print('unique sockets', df['sockets'].unique()[:10])
