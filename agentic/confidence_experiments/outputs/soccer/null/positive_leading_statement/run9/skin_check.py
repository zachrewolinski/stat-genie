import pandas as pd

df = pd.read_csv('soccer.csv')
df['skin'] = df[['rater1','rater2']].mean(axis=1)

player = df.groupby('playerShort', as_index=False).agg(skin=('skin','mean'))
player = player[~player['skin'].isna()]

print('players with skin (groupby mean):', player.shape[0])
print('dark > 0.5 count:', (player['skin'] > 0.5).sum())
print('dark >= 0.5 count:', (player['skin'] >= 0.5).sum())
print('skin value counts (rounded 2):')
print(player['skin'].round(2).value_counts().sort_index())
