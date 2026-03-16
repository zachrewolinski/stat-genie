import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)

# assume games column is 'redCards' (1-47)

games = df['redCards']

candidates = ['yellowCards','meanExp','yellowReds','defeats','player','nIAT','rater2']
for col in candidates:
    s = df[col]
    # check if counts ever exceed games
    exceed = (s > games).mean()
    print(col, 'max', s.max(), 'mean', s.mean(), 'pct_exceed_games', exceed)

