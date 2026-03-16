import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)

# games variable likely 'redCards'
# candidate red card columns
candidates = ['meanExp','yellowCards']

for col in candidates:
    corr = df[col].corr(df['redCards'])
    print(col, 'corr with redCards(games?)', corr)

# correlations with yellowReds (maybe yellow cards)
for col in candidates:
    corr = df[col].corr(df['yellowReds'])
    print(col, 'corr with yellowReds', corr)

# check relationships with player (goals?)
for col in candidates:
    corr = df[col].corr(df['player'])
    print(col, 'corr with player', corr)

# try to infer by rate per game
for col in candidates:
    rate = (df[col] / df['redCards']).replace([np.inf, -np.inf], np.nan)
    print(col, 'mean rate per game', rate.mean(), 'max rate', rate.max())

