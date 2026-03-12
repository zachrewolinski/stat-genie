import pandas as pd

df = pd.read_csv('soccer.csv')

# count unique rater values per player
r1_var = df.groupby('playerShort')['rater1'].nunique(dropna=True)
r2_var = df.groupby('playerShort')['rater2'].nunique(dropna=True)

print('players with >1 unique rater1:', (r1_var > 1).sum())
print('players with >1 unique rater2:', (r2_var > 1).sum())

# show example
example = r1_var[r1_var > 1].index[:5]
print('example players:', list(example))
if len(example) > 0:
    for p in example:
        vals1 = sorted(df.loc[df['playerShort']==p, 'rater1'].dropna().unique())
        vals2 = sorted(df.loc[df['playerShort']==p, 'rater2'].dropna().unique())
        print(p, 'r1', vals1, 'r2', vals2)
