import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nDescribe numeric:')
print(df.describe(include='number'))
print('\nUnique counts:')
for col in df.columns:
    print(col, df[col].nunique())
print('\nSample unique for categorical:')
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, df[col].unique()[:5])
