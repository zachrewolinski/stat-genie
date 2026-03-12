import pandas as pd
import numpy as np

path = 'mortgage.csv'
df = pd.read_csv(path)
print(df.head())
print('shape', df.shape)
for col in df.columns:
    s = df[col]
    uniq = s.dropna().unique()
    nuniq = len(uniq)
    if nuniq<=10:
        print(col, 'unique', sorted(uniq))
    else:
        print(col, 'nunique', nuniq, 'min', s.min(), 'max', s.max(), 'mean', s.mean())
