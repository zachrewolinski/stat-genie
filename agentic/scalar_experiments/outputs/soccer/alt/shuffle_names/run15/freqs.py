import pandas as pd

df = pd.read_csv('soccer.csv')

for c in ['yellowCards','meanExp','yellowReds','redCards']:
    if c in df.columns:
        vc = df[c].value_counts().sort_index()
        print('\n', c, vc.head(20))
