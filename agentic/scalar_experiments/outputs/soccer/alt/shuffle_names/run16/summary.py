import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

for col in df.columns:
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        print(col, 'dtype', s.dtype, 'min', s.min(), 'max', s.max(), 'nunique', s.nunique())
    else:
        print(col, 'dtype', s.dtype, 'nunique', s.nunique())
