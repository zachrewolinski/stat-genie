import pandas as pd
import itertools

path = 'soccer.csv'
df = pd.read_csv(path)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
count_cols = [c for c in num_cols if df[c].max() <= 30 and df[c].min() >= 0 and c not in ['rater1','nExp','position','seIAT','playerShort','refCountry']]
# exclude continuous bias measures etc; keep potential counts
print('count-like', count_cols)

for target in [c for c in num_cols if df[c].max() <= 60 and df[c].min() >= 0]:
    if target in ['rater1','nExp','position','seIAT','playerShort','refCountry']:
        continue
    best = (0, None)
    for combo in itertools.combinations([c for c in count_cols if c != target], 3):
        match = (df[list(combo)].sum(axis=1) == df[target]).mean()
        if match > best[0]:
            best = (match, combo)
    if best[0] > 0.05:
        print(target, 'best match', best)

