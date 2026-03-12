import pandas as pd
import numpy as np

path='mortgage.csv'
df=pd.read_csv(path)

summary=[]
for col in df.columns:
    series=df[col]
    nunique=series.nunique(dropna=True)
    unique_vals=np.sort(series.dropna().unique())
    summary.append((col, series.dtype, nunique, series.isna().mean(), series.min(), series.max(), unique_vals[:5], unique_vals[-5:]))

print('rows', len(df))
print('columns', df.columns.tolist())
print('\nBasic column summary:')
for col, dtype, nunique, na_rate, vmin, vmax, head_vals, tail_vals in summary:
    print(f"{col}: dtype={dtype}, nunique={nunique}, na_rate={na_rate:.3f}, min={vmin}, max={vmax}, head_vals={head_vals}, tail_vals={tail_vals}")

# Identify binary columns
print('\nBinary columns (values subset of {0,1}):')
for col in df.columns:
    vals=set(df[col].dropna().unique())
    if vals.issubset({0,1}):
        print(col, 'mean', df[col].mean(), 'count', df[col].count())

# Crosstab for likely gender vs approve/deny
# We'll try each binary column as potential gender and compute denial rate
binary_cols=[col for col in df.columns if set(df[col].dropna().unique()).issubset({0,1})]
for col in binary_cols:
    if col=='deny':
        continue
    ct=pd.crosstab(df[col], df['deny'], dropna=False)
    # compute denial rate by group
    rates=ct.div(ct.sum(axis=1), axis=0)
    print('\n', col)
    print(ct)
    print('deny rate by', col)
    print(rates[1])

