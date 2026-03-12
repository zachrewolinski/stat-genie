import pandas as pd


df = pd.read_csv('soccer.csv')

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

df = df[df['skin_tone'].notna()]

player = df.groupby('playerShort', as_index=False).agg(skin_tone=('skin_tone', 'mean'))

print(player['skin_tone'].value_counts().sort_index())
print('unique', player['skin_tone'].nunique())
print('min', player['skin_tone'].min(), 'max', player['skin_tone'].max())
print('counts >0.5', (player['skin_tone'] > 0.5).sum())
print('counts >=0.75', (player['skin_tone'] >= 0.75).sum())
