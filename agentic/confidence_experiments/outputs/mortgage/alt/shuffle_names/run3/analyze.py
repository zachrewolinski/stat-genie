import pandas as pd
import numpy as np

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())

# basic stats
summary = []
for col in df.columns:
    s = df[col]
    uniq = s.dropna().unique()
    nuniq = len(uniq)
    minv = s.min()
    maxv = s.max()
    mean = s.mean()
    std = s.std()
    summary.append((col, nuniq, minv, maxv, mean, std))

summary_df = pd.DataFrame(summary, columns=['col','nuniq','min','max','mean','std']).sort_values(['nuniq','mean'])
print('\nSummary sorted by nuniq then mean:')
print(summary_df.to_string(index=False))

# identify binary columns
binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0,1})]
print('\nBinary columns:', binary_cols)

# check complements among binary columns
print('\nComplement pairs (x + y == 1 for all rows):')
for i, c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        if ((df[c1] + df[c2]) == 1).all():
            print(c1, c2)

# Correlation between binary columns
print('\nBinary column means:')
for col in binary_cols:
    print(col, df[col].mean())

# Identify potential outcome variable by strongest correlation with continuous vars (maybe?)
# We'll compute point-biserial correlation with non-binary vars.
nonbinary = [c for c in df.columns if c not in binary_cols]

from scipy.stats import pointbiserialr

print('\nPoint-biserial correlations with each binary column (abs max):')
for b in binary_cols:
    corrs = []
    for c in nonbinary:
        # handle constant
        if df[c].nunique() <= 1:
            continue
        r, p = pointbiserialr(df[b], df[c])
        corrs.append((c, r, p))
    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:3]
    print('\n', b)
    for c, r, p in corrs_sorted:
        print(' ', c, 'r', round(r,3), 'p', p)

