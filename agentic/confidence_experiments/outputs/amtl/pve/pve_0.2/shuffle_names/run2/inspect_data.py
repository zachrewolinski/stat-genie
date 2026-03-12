import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

print(_df.head(12))

# check integer-like columns
for col in _df.columns:
    if _df[col].dtype != 'object':
        frac = np.mean((_df[col] % 1) != 0)
        print(col, 'fraction non-integer', frac)

# describe numeric
print('\nNumeric describe:')
print(_df.describe())

