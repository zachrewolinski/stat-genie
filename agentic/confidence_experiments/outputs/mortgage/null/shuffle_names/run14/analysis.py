import pandas as pd
import numpy as np

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

print('shape', df.shape)
print('columns', df.columns.tolist())

# identify binary columns
binary_cols = []
for c in df.columns:
    vals = df[c].dropna().unique()
    if len(vals) <= 2 and set(np.round(vals,6)).issubset({0,1}):
        binary_cols.append(c)

print('binary cols', binary_cols)
print('binary means')
for c in binary_cols:
    print(c, df[c].mean())

# find complementary pairs
complements = []
for i, c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        if np.allclose(df[c1] + df[c2], 1):
            complements.append((c1, c2))

print('complementary pairs', complements)

# show head
print(df.head())

# show value counts for binary cols
for c in binary_cols:
    print(c, df[c].value_counts().to_dict())

