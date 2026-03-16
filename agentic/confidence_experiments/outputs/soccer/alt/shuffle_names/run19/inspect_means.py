import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

for col in ['defeats','rater2','player','yellowReds','meanExp','yellowCards','redCards']:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        print(col, 'mean', s.mean(), 'median', s.median(), 'max', s.max(), 'min', s.min())
