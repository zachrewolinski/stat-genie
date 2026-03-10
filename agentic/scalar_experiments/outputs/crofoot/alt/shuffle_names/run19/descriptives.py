import pandas as pd

_df = pd.read_csv('crofoot.csv')

# reconstruct sizes and distances
focal_size = _df['f_other']
other_size = _df['win']
rel_size = focal_size - other_size

focal_dist = _df['m_other']
other_dist = _df['n_focal']
rel_loc = focal_dist - other_dist

_df = _df.assign(rel_size=rel_size, rel_loc=rel_loc)

# win rate by size advantage
_df['size_adv'] = pd.cut(_df['rel_size'], bins=[-100, -0.5, 0.5, 100], labels=['focal_smaller', 'equal', 'focal_larger'])
win_rate_size = _df.groupby('size_adv')['m_focal'].mean()
counts_size = _df['size_adv'].value_counts()

# win rate by location advantage (closer to own center)
_df['loc_adv'] = pd.cut(_df['rel_loc'], bins=[-1e9, -1e-6, 1e-6, 1e9], labels=['focal_closer', 'equal', 'focal_farther'])
win_rate_loc = _df.groupby('loc_adv')['m_focal'].mean()
counts_loc = _df['loc_adv'].value_counts()

print('Win rates by size advantage:')
print(win_rate_size)
print('Counts:')
print(counts_size)
print('\nWin rates by location advantage:')
print(win_rate_loc)
print('Counts:')
print(counts_loc)

