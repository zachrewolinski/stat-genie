import pandas as pd


df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]

outcome = 'deny'

for c in binary_cols:
    if c == outcome:
        continue
    rate1 = df.loc[df[c]==1, outcome].mean()
    rate0 = df.loc[df[c]==0, outcome].mean()
    print(c, 'n1', (df[c]==1).sum(), 'approve_rate1', rate1, 'approve_rate0', rate0, 'diff', rate1-rate0)
