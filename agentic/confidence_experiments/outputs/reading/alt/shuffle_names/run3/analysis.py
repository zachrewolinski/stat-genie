import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

# summary of columns
summary = []
for col in df.columns:
    ser = df[col]
    dtype = ser.dtype
    nunique = ser.nunique(dropna=True)
    # sample values
    samples = ser.dropna().unique()[:5]
    summary.append((col, str(dtype), nunique, samples))

for col, dtype, nunique, samples in summary:
    print(f"{col}: dtype={dtype} nunique={nunique} samples={samples}")

# check numeric columns stats
num_cols = df.select_dtypes(include=[np.number]).columns
print('numeric cols', list(num_cols))
print(df[num_cols].describe().T[['min','max','mean','std']])

# identify binary-ish numeric columns
binary_cols = []
for col in num_cols:
    uniq = sorted(df[col].dropna().unique())
    if len(uniq) <= 3 and all(u in [0,1,2] for u in uniq):
        binary_cols.append((col, uniq))
print('binary-ish', binary_cols)

# identify columns that look like reading speed: numeric with range maybe 50-1000
for col in num_cols:
    if df[col].min() > 0 and df[col].max() < 2000:
        print('candidate_speed', col, df[col].min(), df[col].max(), df[col].mean())

