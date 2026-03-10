import pandas as pd

df = pd.read_csv('mortgage.csv')

binary_cols = []
for col in df.columns:
    vals = df[col].dropna().unique()
    if len(vals) <= 3 and set(vals).issubset({0,1}):
        binary_cols.append(col)

for col in binary_cols:
    counts = df[col].value_counts().to_dict()
    print(col, counts, 'mean', df[col].mean())
