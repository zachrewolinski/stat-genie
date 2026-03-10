import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

cand_cols = ['yellowCards','meanExp','yellowReds','defeats','rater2','player']
# identify games column candidate with min>=1 and max moderate
# We'll use column 'redCards' as games based on range 1-47
if 'redCards' in df.columns:
    games_col = 'redCards'
else:
    games_col = None

print('Using games_col:', games_col)

for col in cand_cols:
    if col in df.columns:
        s = df[col]
        print(col, 'mean', s.mean(), 'max', s.max(), 'pct_zero', (s==0).mean())
        if games_col:
            corr = df[[col, games_col]].corr(method='spearman').iloc[0,1]
            print('  spearman with games:', corr)

# examine distribution of candidate red cards by games counts
if games_col:
    for col in ['yellowCards','meanExp']:
        if col in df.columns:
            # compute red card rate per game
            rate = df[col] / df[games_col]
            print('\n', col, 'rate summary (excluding zeros games):')
            print(rate.describe(percentiles=[0.5,0.9,0.95,0.99]))
