import pandas as pd

path='soccer.csv'
df=pd.read_csv(path)
for c in ['rater1','nExp']:
    vals=sorted(df[c].dropna().unique())
    print(c, vals)
