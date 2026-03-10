import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('\ncolumns', df.columns.tolist())

summary = []
for col in df.columns:
    s = df[col]
    dtype = s.dtype
    nunique = s.nunique(dropna=True)
    # get sample values
    samples = s.dropna().unique()[:5]
    summary.append((col, dtype, nunique, samples))

print('\ncolumn summary:')
for col, dtype, nunique, samples in summary:
    print(f"{col}: dtype={dtype}, nunique={nunique}, samples={samples}")

# numeric stats
print('\nNumeric describe:')
print(df.select_dtypes(include=[np.number]).describe().T)

# try identify binary columns
print('\nBinary-like columns:')
for col in df.columns:
    if df[col].dropna().nunique() <= 3:
        print(col, df[col].dropna().unique())

