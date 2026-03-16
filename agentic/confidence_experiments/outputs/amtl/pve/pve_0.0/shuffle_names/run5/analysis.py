import pandas as pd
import json

# Load data

df = pd.read_csv('amtl.csv')

print('shape', df.shape)
print('columns', df.columns.tolist())
print('\nhead')
print(df.head())
print('\nmissing')
print(df.isna().sum())
print('\ndtypes')
print(df.dtypes)

# describe numeric and categorical
print('\ndescribe numeric')
print(df.describe(include='number'))
print('\nvalue counts for categorical (top 5)')
for col in df.columns:
    if df[col].dtype == 'object':
        print('\n', col)
        print(df[col].value_counts().head())
