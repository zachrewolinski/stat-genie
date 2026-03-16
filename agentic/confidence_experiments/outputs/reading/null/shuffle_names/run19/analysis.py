import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)

# Summary of unique counts and sample values per column
summary = []
for col in df.columns:
    series = df[col]
    nunique = series.nunique(dropna=True)
    sample_vals = series.dropna().unique()[:5]
    summary.append((col, nunique, sample_vals))

print('\nUnique counts and samples:')
for col, nunique, sample_vals in summary:
    print(col, nunique, sample_vals)

# numeric describe
print('\nNumeric describe:')
print(df.select_dtypes(include=[np.number]).describe().T[['count','mean','std','min','25%','50%','75%','max']])

