import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

print('Rows', len(df))

# summarize columns
summary=[]
for col in df.columns:
    series=df[col]
    nunique=series.nunique(dropna=False)
    dtype=str(series.dtype)
    sample=series.dropna().unique()[:5]
    summary.append((col,dtype,nunique,sample))

for col,dtype,nunique,sample in summary:
    print(f"{col}: dtype={dtype} nunique={nunique} sample={sample}")

# numeric columns stats
num_cols=df.select_dtypes(include=[np.number]).columns
print('\nNumeric stats:')
for col in num_cols:
    s=df[col]
    print(col, 'min', s.min(), 'max', s.max(), 'mean', s.mean(), 'std', s.std())

# check categorical unique values for object columns
cat_cols=df.select_dtypes(include=['object']).columns
print('\nCategorical unique values:')
for col in cat_cols:
    vals=df[col].dropna().unique()
    print(col, 'unique count', len(vals), 'values', vals[:10])
