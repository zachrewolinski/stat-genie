import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')

# Identify games column candidate (integer, min>=1, max<=60, many unique)
candidates_games = []
for col in _df.columns:
    s = _df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.min() >= 1 and s.max() <= 60 and s.nunique() > 10:
            if np.all(np.mod(s, 1) == 0):
                candidates_games.append(col)
print('games candidates', candidates_games)

# Use redCards as games guess
if 'redCards' in _df.columns:
    games_col = 'redCards'
else:
    games_col = candidates_games[0]

# Check candidate card columns vs games
count_cols = ['yellowCards','yellowReds','defeats','player','nIAT','rater2','meanExp']
print('games_col', games_col)
for col in count_cols:
    if col in _df.columns:
        s = _df[col]
        if pd.api.types.is_numeric_dtype(s):
            print(col, 'max', s.max(), 'min', s.min(), 'pct>games', (s > _df[games_col]).mean())

# list columns with small max <=5 that are <= games
small_cols = []
for col in _df.columns:
    s = _df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.min() >= 0 and s.max() <= 5:
            if (s <= _df[games_col]).all():
                small_cols.append(col)
print('small cols <= games', small_cols)

