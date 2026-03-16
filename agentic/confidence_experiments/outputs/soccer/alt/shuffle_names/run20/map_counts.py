import pandas as pd
import numpy as np
from itertools import combinations


df = pd.read_csv('soccer.csv')
# identify integer count columns (non-negative, mostly integers)
num_cols = df.select_dtypes(include=[np.number]).columns
int_cols = []
for c in num_cols:
    s = df[c]
    if s.min() >= 0 and np.all(np.isclose(s, np.round(s))):
        int_cols.append(c)

print('int cols', int_cols)

# likely games column (max 47, min 1)
possible_games = [c for c in int_cols if df[c].max() >= 30 and df[c].max() <= 60]
print('possible games columns', possible_games)

for target in possible_games:
    cands = [c for c in int_cols if c != target]
    found = False
    for a,b,c in combinations(cands, 3):
        if ((df[a] + df[b] + df[c]) == df[target]).all():
            print('target', target, 'sum match', a,b,c)
            found = True
            break
    if not found:
        # approximate check
        best = None
        for a,b,c in combinations(cands,3):
            diff = (df[a]+df[b]+df[c]-df[target]).abs().sum()
            if best is None or diff < best[0]:
                best = (diff, a,b,c)
        print('target', target, 'best triple diff', best)
