import pandas as pd

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
