import pandas as pd


df = pd.read_csv('soccer.csv')

df = df[df[['rater1','rater2']].notna().any(axis=1)]

# For each player, count unique rater pairs
pairs = df.groupby('playerShort').agg(
    r1_unique=('rater1', lambda x: x.dropna().unique().tolist()),
    r2_unique=('rater2', lambda x: x.dropna().unique().tolist())
).reset_index()

# Find players with multiple unique rater values
multi = pairs[(pairs['r1_unique'].apply(len) > 1) | (pairs['r2_unique'].apply(len) > 1)]
print('players with multiple rater values', len(multi))
print(multi.head(5))

# Check distribution of rater means per player when using first non-null values
player = df.groupby('playerShort').agg(
    rater1=('rater1', lambda x: x.dropna().iloc[0] if x.dropna().size>0 else float('nan')),
    rater2=('rater2', lambda x: x.dropna().iloc[0] if x.dropna().size>0 else float('nan'))
).reset_index()
player['skin_tone'] = player[['rater1','rater2']].mean(axis=1)
print(player['skin_tone'].value_counts().sort_index().head(10))
print('unique skin tones', player['skin_tone'].nunique())
print('counts >0.5', (player['skin_tone'] > 0.5).sum())
print('counts >=0.75', (player['skin_tone'] >= 0.75).sum())
