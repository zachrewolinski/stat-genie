import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('dtypes')
print(df.dtypes)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
print('num_cols', num_cols)
print('cat_cols', cat_cols)
for c in num_cols:
    s = df[c]
    print(c, 'min', s.min(), 'max', s.max(), 'mean', s.mean())

# check which numeric columns are integers
for c in num_cols:
    s = df[c]
    is_int = np.allclose(s, np.round(s))
    print(c, 'all_int', is_int)

# check candidate pairs where col1 <= col2 for most rows
for c1 in num_cols:
    for c2 in num_cols:
        if c1==c2:
            continue
        prop = (df[c1] <= df[c2]).mean()
        if prop > 0.95:
            print('prop', c1, '<=', c2, prop)

# check for columns between 0 and 1
for c in num_cols:
    s = df[c]
    if s.min() >= 0 and s.max() <= 1:
        print('0-1', c)

# examine categories unique values for cat cols
for c in cat_cols:
    print(c, 'nunique', df[c].nunique(), 'sample', df[c].unique()[:5])
