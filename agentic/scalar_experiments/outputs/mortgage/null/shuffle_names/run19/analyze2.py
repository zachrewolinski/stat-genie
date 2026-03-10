import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
print('binary_cols', binary_cols)

# check pairs for complements
pairs = []
for i,c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        s = (df[c1] + df[c2])
        # measure how close to 1
        diff = (s - 1).abs().mean()
        if diff < 0.05:
            pairs.append((c1,c2,diff))

print('near-complements', pairs)

# correlation matrix
print('binary correlation matrix')
print(df[binary_cols].corr())

# Denial-like variable might correlate with credit risk variables? Try logistic for each binary outcome vs some continuous variables
continuous_cols = [c for c in df.columns if c not in binary_cols]
print('continuous_cols', continuous_cols)

# Quick check which binary has prevalence 0.1-0.3
for c in binary_cols:
    print(c, df[c].mean())

