import pandas as pd

df = pd.read_csv('soccer.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
