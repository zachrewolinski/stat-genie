import pandas as pd


df = pd.read_csv('reading.csv')
print(df.head())
print('shape', df.shape)
print('\nunique counts:')
for col in df.columns:
    print(col, df[col].nunique())
print('\nvalue samples:')
for col in df.columns:
    print('\n', col, df[col].dropna().unique()[:10])

