import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
skin_mean = df[['rater1','rater2']].mean(axis=1)
base = df.assign(skin_mean=skin_mean)
base = base[(base['skin_mean'].notna()) & (base['games'] > 0)].copy()

player = (
    base.groupby('playerShort')
    .agg(skin_mean=('skin_mean','mean'),
         total_games=('games','sum'),
         total_red=('redCards','sum'))
    .reset_index()
)

player['skin_group'] = np.where(player['skin_mean'] <= 0.25, 'light', np.where(player['skin_mean'] >= 0.75, 'dark', 'mid'))
print(player['skin_group'].value_counts())
print(player[player['skin_group']=='dark'].head())
