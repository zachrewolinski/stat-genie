import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print('shape', df.shape)
print(df.head())

# summary: unique counts and mean
summary = []
for col in df.columns:
    series = df[col]
    nunique = series.nunique(dropna=False)
    mean = series.mean()
    std = series.std()
    summary.append((col, series.dtype, nunique, mean, std, series.min(), series.max()))

summary_df = pd.DataFrame(summary, columns=['col','dtype','nunique','mean','std','min','max'])
print(summary_df.sort_values('nunique').to_string(index=False))

# detect binary columns
binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print('binary_cols', binary_cols)

# correlations among binary columns
if binary_cols:
    print('binary means')
    for c in binary_cols:
        print(c, df[c].mean())

# compute pairwise correlations with each binary as target
for target in binary_cols:
    corrs = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print('\ncorrelations with', target)
    print(corrs.head(10))
