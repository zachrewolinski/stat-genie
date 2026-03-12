import pandas as pd


df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]
print('binary columns:', binary_cols)
for c in binary_cols:
    print(c, df[c].mean())

# check min/max of non-binary for context
for c in df.columns:
    if c not in binary_cols:
        print(c, df[c].min(), df[c].max(), df[c].mean())
