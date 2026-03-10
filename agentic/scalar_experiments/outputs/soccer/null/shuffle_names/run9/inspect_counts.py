import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
for col in ['yellowCards','meanExp','yellowReds']:
    s = df[col]
    print(col, 'mean', s.mean(), 'max', s.max(), 'zero_frac', (s==0).mean(), 'value_counts_nonzero', s[s>0].value_counts().head())

# correlation with games (redCards)
for col in ['yellowCards','meanExp','yellowReds']:
    print('corr', col, df[col].corr(df['redCards']))
