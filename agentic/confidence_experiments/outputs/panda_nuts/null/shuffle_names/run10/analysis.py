import pandas as pd
import json

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)
print('\nDescribe numeric:')
print(df.describe(include='number'))
print('\nDescribe categorical:')
print(df.describe(include='object'))
