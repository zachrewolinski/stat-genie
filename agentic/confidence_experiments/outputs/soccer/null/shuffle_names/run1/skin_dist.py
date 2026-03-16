import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
skin = df[['rater1','nExp']].mean(axis=1)
player_id = df['photoID']
player_skin = (
    df.assign(skin=skin)
      .dropna(subset=['skin'])
      .groupby(player_id)['skin']
      .mean()
)

print(player_skin.value_counts().sort_index())
print('unique', len(player_skin.unique()))
print('min max', player_skin.min(), player_skin.max())

