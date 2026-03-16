import pandas as pd


df = pd.read_csv('teachingratings.csv')
print(df.head())
print('columns', df.columns.tolist())
print('shape', df.shape)
print(df.describe(include='all').T.head(30))
