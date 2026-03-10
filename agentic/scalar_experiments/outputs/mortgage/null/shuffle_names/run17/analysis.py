import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print(df.head())
print(df.describe(include='all'))
# identify binary columns
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1,0.0,1.0})]
print('binary cols', binary_cols)
for c in binary_cols:
    print(c, df[c].value_counts(dropna=False))
