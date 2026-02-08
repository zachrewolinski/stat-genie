import pandas as pd
import numpy as np

path = 'affairs.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.nunique())
for col in df.columns:
    print('\n', col)
    print(df[col].head(10))
    print('unique sample', df[col].dropna().unique()[:10])
    print('min', df[col].min(), 'max', df[col].max())
    if df[col].dtype == 'object':
        print(df[col].value_counts().head())
