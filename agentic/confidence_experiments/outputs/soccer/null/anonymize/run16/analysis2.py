import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
skin = df[['feature18','feature19']].mean(axis=1)
mask = skin.notna()
use = df.loc[mask].copy()
use['skin_mean'] = skin[mask]

player = use.groupby('feature1', as_index=False).agg(skin_mean=('skin_mean','mean'))

vals = sorted(player['skin_mean'].unique())
print('unique skin_mean values (first 20):', vals[:20])
print('n_unique:', len(vals))
print(player['skin_mean'].describe())

# counts per value
print(player['skin_mean'].value_counts().sort_index())
