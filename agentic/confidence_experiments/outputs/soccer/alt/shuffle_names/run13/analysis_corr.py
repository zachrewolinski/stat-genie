import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

num_cols = df.select_dtypes(include=[np.number]).columns

def corr_with(col):
    s = df[col]
    corrs = {}
    for c in num_cols:
        if c == col:
            continue
        if df[c].nunique() <= 1:
            continue
        corrs[c] = s.corr(df[c])
    return sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

for target in ['redCards','yellowCards','meanExp','yellowReds','defeats','player']:
    if target in df.columns:
        print('\nTop correlations with', target)
        for c, v in corr_with(target)[:8]:
            print(c, v)
