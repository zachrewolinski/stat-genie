import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')
player_col = 'photoID'

# compute for each column percent of players with only one unique value
results = []
for col in _df.columns:
    if col == player_col:
        continue
    nunique_per_player = _df.groupby(player_col)[col].nunique()
    pct_const = (nunique_per_player == 1).mean()
    results.append((col, pct_const, nunique_per_player.mean(), nunique_per_player.max()))

results_sorted = sorted(results, key=lambda x: (-x[1], x[2]))
print('top constant columns')
for col, pct_const, mean_nu, max_nu in results_sorted[:10]:
    print(col, 'pct_const', pct_const, 'mean_nunique', mean_nu, 'max_nunique', max_nu)

print('least constant columns')
for col, pct_const, mean_nu, max_nu in results_sorted[-10:]:
    print(col, 'pct_const', pct_const, 'mean_nunique', mean_nu, 'max_nunique', max_nu)

