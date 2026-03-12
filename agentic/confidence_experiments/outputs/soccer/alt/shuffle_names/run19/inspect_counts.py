import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if len(vals)==0:
            continue
        # check if integer-like
        if np.all(np.isclose(vals, np.round(vals))):
            print(col, 'min', int(vals.min()), 'max', int(vals.max()), 'n_unique', vals.nunique())
