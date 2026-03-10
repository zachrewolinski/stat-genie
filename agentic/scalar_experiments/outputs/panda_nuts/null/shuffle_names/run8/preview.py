import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nDescribe:')
print(df.describe(include='all'))
print('\nUnique values:')
for col in df.columns:
    print(col, df[col].nunique(), df[col].head(5).tolist())
