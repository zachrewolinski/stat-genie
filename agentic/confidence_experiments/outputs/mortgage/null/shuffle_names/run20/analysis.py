import pandas as pd
import numpy as np

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)

# basic stats for candidate gender column
for col in df.columns:
    if df[col].dropna().isin([0,1]).all():
        uniq = df[col].dropna().unique()
        if len(uniq)<=2:
            print('binary', col, df[col].mean())

# check accept/deny columns
for col in ['accept','deny','female']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head())

# cross-tab female vs deny/accept
if 'female' in df.columns:
    if 'deny' in df.columns:
        print('ct female vs deny')
        print(pd.crosstab(df['female'], df['deny']))
    if 'accept' in df.columns:
        print('ct female vs accept')
        print(pd.crosstab(df['female'], df['accept']))

