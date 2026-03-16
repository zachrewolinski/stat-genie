import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')
# games column identified as redCards

games = df['redCards']

candidates = ['meanExp','yellowCards','yellowReds','player']
for c in candidates:
    corr = games.corr(df[c])
    print(c, 'corr with games', corr)

# show distribution counts for candidates
for c in candidates:
    print('\n', c, 'value counts')
    print(df[c].value_counts().sort_index())
