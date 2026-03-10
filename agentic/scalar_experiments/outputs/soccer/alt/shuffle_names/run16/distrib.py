import pandas as pd
import numpy as np

cols = ['redCards','nIAT','defeats','rater2','player','yellowReds','meanExp','yellowCards']
df = pd.read_csv('soccer.csv')

for col in cols:
    s = df[col]
    print('\n', col)
    print(s.describe())
    print('pct_zero', (s==0).mean())
    print('pct_one', (s==1).mean())
    print('pct_gt1', (s>1).mean())
