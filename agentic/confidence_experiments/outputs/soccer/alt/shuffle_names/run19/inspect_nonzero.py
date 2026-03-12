import pandas as pd

df = pd.read_csv('soccer.csv')
for col in ['meanExp','yellowCards']:
    s = df[col]
    nonzero = (s>0).sum()
    print(col, 'nonzero', nonzero, 'pct', nonzero/len(df))
    print(s.value_counts().sort_index().head(10))
