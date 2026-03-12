import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('amtl.csv')
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)

# Unique values for each column
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col} unique (sample):", df[col].unique()[:10])
    else:
        print(f"\n{col} stats: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}, n_unique={df[col].nunique()}")
