import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

print('shape', df.shape)
print('columns', list(df.columns))

# summarize each column: dtype, unique count, sample values, min/max if numeric
summary = []
for col in df.columns:
    s = df[col]
    nuniq = s.nunique(dropna=True)
    dtype = s.dtype
    sample = s.dropna().unique()[:5]
    row = {'col': col, 'dtype': str(dtype), 'nuniq': nuniq, 'sample': sample}
    if pd.api.types.is_numeric_dtype(s):
        row.update({
            'min': float(np.nanmin(s)),
            'max': float(np.nanmax(s)),
            'mean': float(np.nanmean(s)),
            'std': float(np.nanstd(s))
        })
    summary.append(row)

# Print summary in order of columns
for row in summary:
    print(row)
