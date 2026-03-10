import pandas as pd

path = 'mortgage.csv'
df = pd.read_csv(path)

binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print('binary cols', binary_cols)
for c in binary_cols:
    print(c, 'mean', df[c].mean())
