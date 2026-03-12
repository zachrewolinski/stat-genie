import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
for c in ['yellowCards','meanExp','yellowReds','redCards']:
    if c in df.columns:
        s = df[c]
        print(c, 'min', s.min(), 'max', s.max(), 'mean', s.mean(), 'nunique', s.nunique())
        print(s.value_counts().head(10))
        print('---')
