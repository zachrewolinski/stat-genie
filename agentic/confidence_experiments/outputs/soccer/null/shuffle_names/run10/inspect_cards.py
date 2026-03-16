import pandas as pd
import numpy as np

path='soccer.csv'

df=pd.read_csv(path)

for col in ['yellowCards','meanExp','yellowReds','redCards']:
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        print(col, 'max', s.max(), 'mean', s.mean(), 'pct>0', (s>0).mean())

