import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

for c in df.columns:
    n = df[c].nunique(dropna=False)
    if n <= 10:
        vals = sorted(df[c].dropna().unique().tolist())
        print(c, n, vals)
