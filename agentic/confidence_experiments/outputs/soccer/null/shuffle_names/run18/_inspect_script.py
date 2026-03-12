import pandas as pd
import json

csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.describe(include='all').transpose().head(20))
