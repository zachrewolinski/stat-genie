import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

print('rows', len(df))
print('rater1 unique', sorted(df['rater1'].dropna().unique())[:10], 'n', df['rater1'].nunique(dropna=True))
print('rater2 unique', sorted(df['rater2'].dropna().unique())[:10], 'n', df['rater2'].nunique(dropna=True))

print('rater1 missing', df['rater1'].isna().mean())
print('rater2 missing', df['rater2'].isna().mean())

# compute skin per row
skin = df[['rater1','rater2']].mean(axis=1)
print('skin unique sample', sorted(skin.dropna().unique())[:10], 'n', skin.nunique(dropna=True))

# aggregate per player
player_df = df.assign(skin=skin).groupby('playerShort', as_index=False).agg(
    games=('games','sum'),
    redCards=('redCards','sum'),
    skin=('skin','mean')
)

print('players', len(player_df))
print('player skin missing', player_df['skin'].isna().mean())
print('player skin unique n', player_df['skin'].nunique(dropna=True))

# distribution bins
bins = [0,0.25,0.5,0.75,1.0]
labels = ['<=0.25','0.25-0.5','0.5-0.75','0.75-1']
player_df_non = player_df.dropna(subset=['skin'])
player_df_non['bin'] = pd.cut(player_df_non['skin'], bins=bins, include_lowest=True, right=True, labels=labels)
print(player_df_non['bin'].value_counts().sort_index())

# unique skin values
print('skin value counts (top 10)')
print(player_df_non['skin'].value_counts().head(10))
