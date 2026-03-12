import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
row_skin = df[['rater1','rater2']].mean(axis=1)
player_skin = (
    df.assign(row_skin=row_skin)
      .groupby('playerShort')['row_skin']
      .mean()
)
print('players with skin', player_skin.notna().sum())
print(player_skin.value_counts().sort_index())

# show min/max
print('min', player_skin.min(), 'max', player_skin.max())

# counts in bins
bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
labels = ['<=0.25', '(0.25,0.5]', '(0.5,0.75]', '>0.75']
print(pd.cut(player_skin, bins=bins, labels=labels).value_counts().sort_index())
