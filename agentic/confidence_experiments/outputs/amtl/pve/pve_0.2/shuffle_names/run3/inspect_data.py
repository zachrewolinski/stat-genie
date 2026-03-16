import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
num_cols = df.select_dtypes(include='number').columns
print('num cols', list(num_cols))
for col in num_cols:
    vals = df[col]
    print('\n', col)
    print('min', vals.min(), 'max', vals.max(), 'mean', vals.mean(), 'std', vals.std())
    frac = (vals - vals.round()).abs().mean()
    print('avg abs frac', frac)
    uniq = np.unique(vals)
    print('unique count', len(uniq))
    print('first 10 uniq', uniq[:10])
    print('last 10 uniq', uniq[-10:])

print('\nCategorical counts:')
for col in df.select_dtypes(exclude='number').columns:
    print(col, df[col].unique()[:10])
    print('nunique', df[col].nunique())
