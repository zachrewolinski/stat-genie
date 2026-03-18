import pandas as pd
import json

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.dtypes)
print('n rows', len(df))
for col in df.columns:
    series = df[col]
    print('\n', col)
    print(series.describe(include='all'))
    print('unique sample', series.dropna().unique()[:10])
