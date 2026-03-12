import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

# identify integer-like columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# candidate columns with small integer ranges (<=50) for games/victories etc
candidates = [c for c in num_cols if df[c].max() <= 50]
print("candidates <=50:", candidates)

# Suppose games column has max ~47 -> find column with max >40 and <=50
for c in candidates:
    if df[c].max() >= 40:
        print("possible games", c, df[c].min(), df[c].max(), df[c].mean())

# Check combinations for victories/ties/defeats that sum to games column
# We'll test each triple from candidates excluding games.

import itertools

# choose games candidate with max>40
possible_games = [c for c in candidates if df[c].max() >= 40]

for g in possible_games:
    cols = [c for c in candidates if c != g]
    best = []
    for a,b,c in itertools.combinations(cols,3):
        diff = (df[a] + df[b] + df[c] - df[g]).abs()
        # consider if most rows match exactly
        match = (diff == 0).mean()
        if match > 0.9:
            best.append((match, (a,b,c)))
    best.sort(reverse=True)
    print("\nGames candidate", g, "top matches", best[:5])

# also check if any two sum to games (maybe games not included?)
for g in possible_games:
    cols = [c for c in candidates if c != g]
    best = []
    for a,b in itertools.combinations(cols,2):
        diff = (df[a] + df[b] - df[g]).abs()
        match = (diff == 0).mean()
        if match > 0.9:
            best.append((match, (a,b)))
    best.sort(reverse=True)
    print("\nGames candidate", g, "top 2-sum matches", best[:5])
