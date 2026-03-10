import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if df[c].nunique() == 2]

print('Binary columns:', binary_cols)

# check complements
for i, col1 in enumerate(binary_cols):
    for col2 in binary_cols[i+1:]:
        if (df[col1] == 1 - df[col2]).mean() > 0.95:
            print('near complement', col1, col2, (df[col1] == 1 - df[col2]).mean())
        if (df[col1] == df[col2]).mean() > 0.95:
            print('near same', col1, col2, (df[col1] == df[col2]).mean())

# correlation matrix
corr = df[binary_cols].corr()
print('\nCorrelation matrix:')
print(corr)

