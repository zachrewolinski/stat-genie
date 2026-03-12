import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
skin = df[['rater1','rater2']].mean(axis=1)
df = df.assign(skin_mean=skin)

# count unique skin_mean per player
skin_unique = df.groupby('playerShort')['skin_mean'].nunique(dropna=True)
print('players total', df['playerShort'].nunique())
print('players with >1 skin_mean', (skin_unique > 1).sum())
print('max unique skin_mean per player', skin_unique.max())

# show some players with multiple values
multi = skin_unique[skin_unique > 1]
print('example players', multi.head().index.tolist())

# check if rater1/2 missing patterns vary
missing_by_player = df.groupby('playerShort')['skin_mean'].apply(lambda s: s.isna().mean())
print('players with some missing skin_mean', (missing_by_player.between(0,1, inclusive='neither')).sum())

# show distribution of skin_mean values
print('skin_mean unique values', sorted(df['skin_mean'].dropna().unique())[:10], '... total', df['skin_mean'].nunique())

# count skin groups per player
bins = np.where(df['skin_mean']<=0.25, 'light', np.where(df['skin_mean']>=0.75, 'dark', 'mid'))
df['skin_group'] = bins
player_groups = df.groupby('playerShort')['skin_group'].nunique()
print('players with >1 skin_group', (player_groups>1).sum())
print('max skin_group per player', player_groups.max())
