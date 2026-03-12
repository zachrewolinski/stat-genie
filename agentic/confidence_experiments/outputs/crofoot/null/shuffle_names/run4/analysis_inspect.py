import pandas as pd

path = 'crofoot.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nSummary:')
print(df.describe(include='all'))
print('\nUnique counts:')
for c in df.columns:
    vals = sorted(df[c].unique())
    print(c, df[c].nunique(), vals[:20])
