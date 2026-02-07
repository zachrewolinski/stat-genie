import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)

# basic summaries
for col in df.columns:
    if df[col].dtype == object:
        print(f"\n{col} unique sample:", df[col].dropna().unique()[:10])
        print('nunique', df[col].nunique())
    else:
        print(f"\n{col} summary:")
        print(df[col].describe())

