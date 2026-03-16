import pandas as pd

df = pd.read_csv('panda_nuts.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nUnique counts:')
print(df.nunique())
print('\nUnique values for categorical-like columns:')
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, sorted(df[col].unique()))

print('\nSummary numeric:')
print(df.describe())
