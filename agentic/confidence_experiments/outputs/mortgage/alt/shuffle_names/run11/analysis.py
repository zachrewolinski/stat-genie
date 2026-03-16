import pandas as pd
import numpy as np

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

# summary for each column
summary = []
for col in df.columns:
    s = df[col]
    nuniq = s.nunique(dropna=True)
    summary.append((col, nuniq, s.min(), s.max(), s.mean()))

for col, nuniq, mn, mx, mean in summary:
    if nuniq <= 10:
        uniq = sorted(df[col].dropna().unique().tolist())
        print(col, 'unique', uniq)
    else:
        print(col, 'nunique', nuniq, 'min', mn, 'max', mx, 'mean', mean)
