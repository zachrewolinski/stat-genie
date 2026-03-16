import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
combos = [(['nIAT','defeats','rater2'], 'redCards'),
          (['nIAT','defeats','rater2'], 'player'),
          (['nIAT','defeats','rater2'], 'yellowReds'),
          (['nIAT','defeats','rater2'], 'meanExp'),
          (['nIAT','defeats','rater2'], 'yellowCards')]
for cols, target in combos:
    match = (df[target] == df[cols].sum(axis=1)).mean()
    print(target, cols, match)
