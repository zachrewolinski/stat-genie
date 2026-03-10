import pandas as pd


df = pd.read_csv('soccer.csv')
df['skin_tone'] = df[['rater1','rater2']].mean(axis=1)
player = (
    df[['playerShort','skin_tone']]
    .dropna()
    .groupby('playerShort', as_index=False)
    .agg(skin_tone=('skin_tone','mean'))
)
print(player['skin_tone'].describe())
print(player['skin_tone'].value_counts().sort_index())
print('>=0.5:', (player['skin_tone']>=0.5).sum())
