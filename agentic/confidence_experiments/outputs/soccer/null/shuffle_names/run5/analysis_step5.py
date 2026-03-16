import pandas as pd

_df = pd.read_csv('soccer.csv')
for player_col in ['photoID','goals']:
    for col in ['rater1','nExp']:
        nunique_per_player = _df.groupby(player_col)[col].nunique()
        print('player_col', player_col, 'col', col, 'pct_const', (nunique_per_player==1).mean(), 'mean_nunique', nunique_per_player.mean(), 'max_nunique', nunique_per_player.max())

