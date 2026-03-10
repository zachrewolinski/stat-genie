import pandas as pd
import numpy as np

_df = pd.read_csv('crofoot.csv')
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']

# Win rates by relative size sign
bins = {
    'focal_smaller': _df[_df['rel_size'] < 0],
    'equal_size': _df[_df['rel_size'] == 0],
    'focal_larger': _df[_df['rel_size'] > 0],
}
print('Win rates by relative size:')
for k, d in bins.items():
    if len(d) == 0:
        print(k, 'n=0')
    else:
        print(k, f"n={len(d)} win_rate={d['win'].mean():.3f}")

# Win rates by closer-to-center (rel_location > 0)
closer = _df[_df['rel_location'] > 0]
farther = _df[_df['rel_location'] < 0]
zero = _df[_df['rel_location'] == 0]
print('\nWin rates by location advantage:')
for label, d in [('focal_closer', closer), ('focal_farther', farther), ('equal_distance', zero)]:
    if len(d) == 0:
        print(label, 'n=0')
    else:
        print(label, f"n={len(d)} win_rate={d['win'].mean():.3f}")
