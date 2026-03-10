import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

binary_cols = []
for col in df.columns:
    if df[col].dropna().nunique() == 2:
        binary_cols.append(col)

ternary_cols = []
for col in df.columns:
    if df[col].dropna().nunique() == 3 and pd.api.types.is_numeric_dtype(df[col]):
        ternary_cols.append(col)

print('binary cols', binary_cols)
print('ternary cols', ternary_cols)

for tcol in ternary_cols:
    t = df[tcol]
    indicator = (t > 0).astype(int)
    for bcol in binary_cols:
        b = df[bcol]
        if b.dtype == object:
            uniq = list(pd.unique(b.dropna()))
            if len(uniq) != 2:
                continue
            mapping = {uniq[0]:0, uniq[1]:1}
            bnum = b.map(mapping)
        else:
            bnum = b
        bnum = bnum.fillna(0).astype(int)
        match = (bnum == indicator).mean()
        corr = bnum.corr(indicator)
        print(f'{tcol} indicator vs {bcol}: match {match:.3f} corr {corr:.3f}')
    print('---')
