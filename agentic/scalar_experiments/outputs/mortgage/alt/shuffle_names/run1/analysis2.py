import pandas as pd
import numpy as np

_df = pd.read_csv('mortgage.csv')

summary = []
for c in _df.columns:
    vals = _df[c].dropna().unique()
    summary.append((c, len(vals), np.min(vals), np.max(vals)))
summary_sorted = sorted(summary, key=lambda x: x[1])
print('unique counts (sorted):')
for c, n, vmin, vmax in summary_sorted:
    print(c, n, vmin, vmax)

# columns with 2-3 unique values
print('\ncolumns with <=3 unique values:')
for c, n, vmin, vmax in summary_sorted:
    if n <= 3:
        print(c, 'n', n, 'values', sorted(_df[c].dropna().unique()))

