import pandas as pd
import json

# Load data
df = pd.read_csv('reading.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
