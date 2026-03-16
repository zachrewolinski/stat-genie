import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('soccer.csv')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# candidate integer count columns (non-negative, smallish max)
count_cols = [c for c in num_cols if df[c].min() >= 0 and df[c].max() <= 50]
print('count_cols', count_cols)

# find columns that equal sum of three others frequently
results = []
for target in count_cols:
    t = df[target]
    for combo in combinations([c for c in count_cols if c != target], 3):
        s = df[list(combo)].sum(axis=1)
        match = (t == s).mean()
        if match > 0.9:
            results.append((target, combo, match))

results.sort(key=lambda x: -x[2])
print('sum-of-3 matches >0.9', results[:20])

# also check any column equals sum of two others
results2 = []
for target in count_cols:
    t = df[target]
    for combo in combinations([c for c in count_cols if c != target], 2):
        s = df[list(combo)].sum(axis=1)
        match = (t == s).mean()
        if match > 0.9:
            results2.append((target, combo, match))
results2.sort(key=lambda x: -x[2])
print('sum-of-2 matches >0.9', results2[:20])

