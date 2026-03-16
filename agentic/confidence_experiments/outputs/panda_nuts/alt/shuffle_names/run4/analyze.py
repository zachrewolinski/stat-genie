import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('\ninfo')
print(df.dtypes)
print('\nunique counts')
for col in df.columns:
    print(col, df[col].nunique())
print('\nvalue samples')
for col in df.columns:
    vals = df[col].dropna().unique()[:10]
    print(col, vals)

