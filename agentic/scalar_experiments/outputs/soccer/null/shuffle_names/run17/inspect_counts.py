import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

# redCards column currently is games? We'll use it as games proxy for now
for col in ['yellowCards','meanExp','yellowReds','defeats','rater2','player','nIAT','redCards']:
    s=df[col]
    print(col, 'min', s.min(), 'max', s.max(), 'mean', s.mean(), 'zero%', (s==0).mean())

print('\ncorrelation with redCards (current column)')
for col in ['yellowCards','meanExp','yellowReds','defeats','rater2','player','nIAT']:
    corr = df[[col,'redCards']].corr().iloc[0,1]
    print(col, corr)

