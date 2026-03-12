import pandas as pd
import itertools

path = 'soccer.csv'
df = pd.read_csv(path)

# candidate numeric columns with max <= 30 (likely counts)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
small_cols = [c for c in num_cols if df[c].max() <= 30]
print('small cols', small_cols)

# choose target: column with max 47 (redCards) or maybe something else
# identify column with max between 40 and 60
candidates = [c for c in num_cols if 40 <= df[c].max() <= 60]
print('max 40-60', candidates)

if candidates:
    target = candidates[0]
else:
    target = None
print('target', target)

if target:
    # check combinations of 3 small cols summing to target
    best = []
    for combo in itertools.combinations([c for c in small_cols if c != target], 3):
        s = df[list(combo)].sum(axis=1)
        match = (s == df[target]).mean()
        if match > 0.5:
            best.append((match, combo))
    best.sort(reverse=True)
    print('top combos')
    for m, combo in best[:10]:
        print(m, combo)

