import pandas as pd
import itertools


df = pd.read_csv('soccer.csv')

int_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# keep integer-like columns
import numpy as np
int_cols = [c for c in int_cols if np.all(np.isclose(df[c].dropna(), np.round(df[c].dropna())))]

print('int cols', int_cols)

results = []
for target in int_cols:
    others = [c for c in int_cols if c != target]
    best = (0, None)
    for combo in itertools.combinations(others, 3):
        match = (df[list(combo)].sum(axis=1) == df[target]).mean()
        if match > best[0]:
            best = (match, combo)
    results.append((target, best[0], best[1]))

# sort by best match
results.sort(key=lambda x: x[1], reverse=True)
for target, match, combo in results[:10]:
    print(target, match, combo)
