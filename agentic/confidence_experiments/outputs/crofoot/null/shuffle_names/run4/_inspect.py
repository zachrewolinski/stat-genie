import pandas as pd
import json

path = 'crofoot.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nSummary:')
print(df.describe(include='all'))
print('\nUnique counts:')
for c in df.columns:
    print(c, df[c].nunique(), sorted(df[c].unique())[:10])
