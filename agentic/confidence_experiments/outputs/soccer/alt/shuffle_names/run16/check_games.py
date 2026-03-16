import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
cols = ['nIAT','defeats','rater2','player','yellowReds','meanExp','yellowCards','redCards']

# try combos of 3 among candidate count columns to see if sum equals redCards for many rows
from itertools import combinations

cands = ['nIAT','defeats','rater2','player','yellowReds','meanExp','yellowCards']

for comb in combinations(cands, 3):
    total = df[list(comb)].sum(axis=1)
    match = (total == df['redCards']).mean()
    if match > 0.5:
        print('comb', comb, 'match', match)

# also check if any column equals redCards
for col in cands:
    match = (df[col] == df['redCards']).mean()
    if match > 0.5:
        print('col', col, 'match', match)

