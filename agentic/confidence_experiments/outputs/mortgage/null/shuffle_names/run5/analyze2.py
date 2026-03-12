import pandas as pd
import numpy as np

_df = pd.read_csv('mortgage.csv')

cols = _df.columns.tolist()
# Check exact matches between columns
matches = []
for i,c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        s1 = _df[c1]
        s2 = _df[c2]
        if s1.equals(s2):
            matches.append((c1,c2))

print('Exact matching columns:', matches)

# Check high correlation for binary
for c in cols:
    if _df[c].dropna().nunique() <= 2:
        for d in cols:
            if c==d: continue
            if _df[d].dropna().nunique() <= 2:
                # compute agreement
                a = (_df[c] == _df[d]).mean()
                if a > 0.95:
                    print('High agreement', c, d, a)

# Show means for binary columns
print('\nBinary column means')
for c in cols:
    if _df[c].dropna().nunique() <= 2:
        print(c, _df[c].mean())
