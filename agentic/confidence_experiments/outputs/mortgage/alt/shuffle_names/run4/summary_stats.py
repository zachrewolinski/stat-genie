import pandas as pd
import numpy as np

df=pd.read_csv('mortgage.csv')

summary=[]
for c in df.columns:
    s=df[c]
    summary.append({
        'col':c,
        'min':s.min(),
        'max':s.max(),
        'mean':s.mean(),
        'nunique':s.nunique(dropna=True),
        'missing':s.isna().sum()
    })

# sort by unique count then range
summary=sorted(summary, key=lambda x: (x['nunique'], x['max']))
for row in summary:
    print(row)

