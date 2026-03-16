import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

# pairwise correlation and sums
cols = ['accept','self_employed','deny']
for c in cols:
    if c in df.columns:
        print(c, 'mean', df[c].mean(), 'unique', df[c].unique())

for a in cols:
    for b in cols:
        if a>=b: continue
        s = df[a] + df[b]
        print(a,b,'sum unique', sorted(s.unique()))
        print('corr', np.corrcoef(df[a], df[b])[0,1])

# check if deny is 1-accept for any pair
for a in cols:
    for b in cols:
        if a==b: continue
        diff = (df[a] == 1 - df[b]).mean()
        print(a,'==1-',b, diff)

