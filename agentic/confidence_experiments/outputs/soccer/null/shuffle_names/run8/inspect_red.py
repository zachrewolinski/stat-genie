import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')
# candidate columns for red cards
candidates = ['yellowCards','meanExp','yellowReds']
# games column suspect: redCards
print('games col redCards stats', df['redCards'].describe())
for c in candidates:
    print('\n', c)
    print(df[c].describe())
    print('zeros', (df[c]==0).mean())
    # correlation with games
    corr = np.corrcoef(df['redCards'], df[c])[0,1]
    print('corr with games', corr)

# also check other columns maybe red cards?
for c in ['defeats','player','nIAT','rater2']:
    if c in df.columns:
        corr = np.corrcoef(df['redCards'], df[c])[0,1]
        print(c, 'corr with games', corr, 'mean', df[c].mean(), 'max', df[c].max())
