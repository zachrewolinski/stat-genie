import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)

for col in ['yellowCards','meanExp','yellowReds','defeats','player','nIAT','rater2','redCards']:
    s = df[col]
    print(col, 'min', s.min(), 'max', s.max(), 'mean', s.mean(), 'median', s.median(), 'pct_zero', (s==0).mean())

print('\nvalue counts for yellowCards:')
print(df['yellowCards'].value_counts().sort_index())
print('\nvalue counts for meanExp:')
print(df['meanExp'].value_counts().sort_index())
print('\nvalue counts for redCards:')
print(df['redCards'].value_counts().sort_index().head())

