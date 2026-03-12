import sys, os
for p in ["", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)
import pandas as pd

df = pd.read_csv('soccer.csv')
skin = df[['rater1','rater2']].mean(axis=1)
df = df.assign(skin=skin)

var = df.groupby('playerShort')['skin'].nunique(dropna=True).reset_index(name='n_unique')
print('players with >1 unique skin values:', (var['n_unique']>1).sum())
print('max unique values', var['n_unique'].max())

# show examples
examples = var[var['n_unique']>1].head(5)['playerShort'].tolist()
print('examples', examples)
if examples:
    for p in examples:
        vals = df.loc[df['playerShort']==p, 'skin'].dropna().unique()
        print(p, sorted(vals))
