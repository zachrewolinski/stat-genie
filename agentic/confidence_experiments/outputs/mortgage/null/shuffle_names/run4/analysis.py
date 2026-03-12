import pandas as pd
import numpy as np

path = "mortgage.csv"

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all').T)

# quick checks for binary columns
binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print("binary_cols", binary_cols)

# check correlations between candidate accept/deny columns
for a in ['deny','self_employed','accept','denied_PMI','female']:
    if a in df.columns:
        print(a, df[a].value_counts(dropna=False).head())

# compute relation between deny and self_employed
if 'deny' in df.columns and 'self_employed' in df.columns:
    print('deny vs self_employed crosstab')
    print(pd.crosstab(df['deny'], df['self_employed']))

# check if any pair are perfect inverse
for c1 in df.columns:
    if set(df[c1].dropna().unique()).issubset({0,1}):
        for c2 in df.columns:
            if c1>=c2: # avoid duplicates
                continue
            if set(df[c2].dropna().unique()).issubset({0,1}):
                # check if c1 == 1 - c2
                if np.allclose(df[c1], 1-df[c2]):
                    print("inverse", c1, c2)

