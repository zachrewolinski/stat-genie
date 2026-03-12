import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

binary_cols = []
for col in df.columns:
    vals = df[col].dropna().unique()
    if set(vals).issubset({0, 1}):
        binary_cols.append(col)

print('binary_cols', binary_cols)
for col in binary_cols:
    print(col, 'mean', df[col].mean(), 'count', df[col].count())

# check accept + deny relationship
print('accept+deny value_counts')
print((df['accept'] + df['deny']).value_counts().head())
print('accept==1 & deny==1', ((df['accept'] == 1) & (df['deny'] == 1)).sum())
print('accept==0 & deny==0', ((df['accept'] == 0) & (df['deny'] == 0)).sum())

# correlations with deny
print('correlations with deny (binary cols)')
for col in binary_cols:
    if col == 'deny':
        continue
    corr = np.corrcoef(df[col].fillna(0), df['deny'])[0, 1]
    print(col, corr)

