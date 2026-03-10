import pandas as pd
import json

csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
