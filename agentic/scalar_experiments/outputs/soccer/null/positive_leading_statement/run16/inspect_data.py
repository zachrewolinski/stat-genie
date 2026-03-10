import sys, os
# Remove current directory from sys.path to avoid shadowing stdlib 'inspect'
for p in ["", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)

import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

print('rows', len(df))
print('rater1 unique', sorted(df['rater1'].dropna().unique()), 'n', df['rater1'].nunique(dropna=True))
print('rater2 unique', sorted(df['rater2'].dropna().unique()), 'n', df['rater2'].nunique(dropna=True))

print('rater1 missing', df['rater1'].isna().mean())
print('rater2 missing', df['rater2'].isna().mean())

skin = df[['rater1','rater2']].mean(axis=1)
print('skin unique', sorted(skin.dropna().unique()), 'n', skin.nunique(dropna=True))

player_df = df.assign(skin=skin).groupby('playerShort', as_index=False).agg(
    games=('games','sum'),
    redCards=('redCards','sum'),
    skin=('skin','mean')
)

print('players', len(player_df))
print('player skin missing', player_df['skin'].isna().mean())
print('player skin unique n', player_df['skin'].nunique(dropna=True))

bins = [0,0.25,0.5,0.75,1.0]
labels = ['<=0.25','0.25-0.5','0.5-0.75','0.75-1']
player_df_non = player_df.dropna(subset=['skin']).copy()
player_df_non['bin'] = pd.cut(player_df_non['skin'], bins=bins, include_lowest=True, right=True, labels=labels)
print('bin counts:')
print(player_df_non['bin'].value_counts().sort_index())

print('skin value counts (top 10):')
print(player_df_non['skin'].value_counts().head(10))
