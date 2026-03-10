import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')
skin = df[['feature18', 'feature19']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# unique values and counts
counts = df['skin_tone'].value_counts(dropna=False).sort_index()
print(counts.head(20))
print('unique values:', len(counts))

# player-level unique skin values
player_skin = df.dropna(subset=['skin_tone']).groupby('feature1')['skin_tone'].mean()
player_counts = player_skin.value_counts().sort_index()
print(player_counts)

# Quantiles
for q in [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9]:
    print('q', q, player_skin.quantile(q))

