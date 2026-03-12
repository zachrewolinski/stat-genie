import pandas as pd

_df = pd.read_csv('soccer.csv')
player_col = 'photoID'
for col in ['rater1','nExp','weight','height','ties','victories','seExp']:
    nunique_per_player = _df.groupby(player_col)[col].nunique()
    print(col, 'pct_const', (nunique_per_player==1).mean(), 'mean_nunique', nunique_per_player.mean(), 'max_nunique', nunique_per_player.max())

