import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('rows', len(df), 'cols', df.shape[1])
print(df.columns.tolist())

# summarize numeric columns
summary = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        summary.append((col, s.min(), s.max(), s.nunique(dropna=True), s.isna().mean()))

print('\nNumeric summary (col, min, max, nunique, nan_rate):')
for row in summary:
    print(row)

# Show value counts for columns with few unique values
print('\nLow-unique columns (<=10 unique):')
for col in df.columns:
    nun = df[col].nunique(dropna=True)
    if nun <= 10:
        print(col, 'nunique', nun, 'sample counts', df[col].value_counts(dropna=False).head().to_dict())

# Show correlations with potential red card column candidates (low integer counts)
# Identify candidate card columns: integer-like small counts
candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.dropna().apply(float.is_integer).all() and s.max() <= 50:
            candidates.append(col)
print('\nInteger small-count candidates:', candidates)

