import pandas as pd


df = pd.read_csv('teachingratings.csv')
print('columns', df.columns.tolist())
print(df.head())
print(df.describe(include='all').T)
