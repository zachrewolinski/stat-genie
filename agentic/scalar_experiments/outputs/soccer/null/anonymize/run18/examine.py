import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
df['skin_tone'] = df[['feature18','feature19']].mean(axis=1, skipna=True)
player = df.groupby('feature1', as_index=False).agg(skin_tone=('skin_tone','mean'), games=('feature9','sum'), red_cards=('feature16','sum'))
player = player[~player['skin_tone'].isna()]
print('players with skin tone', len(player))
print(player['skin_tone'].describe())
print('unique skin_tone values (top 20):')
print(player['skin_tone'].value_counts().head(20))

# counts by bins
bins = [0,0.25,0.5,0.75,1.0]
labels = ['0-0.25','0.25-0.5','0.5-0.75','0.75-1.0']
player['bin'] = pd.cut(player['skin_tone'], bins=bins, labels=labels, include_lowest=True, right=True)
print(player['bin'].value_counts(dropna=False))

# counts >0.5
print('>0.5', (player['skin_tone']>0.5).sum())
print('>=0.5', (player['skin_tone']>=0.5).sum())
print('==0.5', (player['skin_tone']==0.5).sum())
