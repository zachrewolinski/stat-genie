import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Skin tone score (average of raters)
_df['skin_tone'] = _df[['rater1', 'rater2']].mean(axis=1)

# Red card rate per game
_df['red_per_game'] = _df['redCards'] / _df['games']

# Median split for descriptive comparison
median_skin = _df['skin_tone'].median()
_df['dark'] = (_df['skin_tone'] > median_skin).astype(int)

# Descriptive rates (dyad-level)
rate_by_group = _df.groupby('dark')['red_per_game'].mean()

# Poisson regression with exposure offset (dyad-level)
poisson_dyad = sm.GLM(
    _df['redCards'],
    sm.add_constant(_df['skin_tone']),
    family=sm.families.Poisson(),
    offset=np.log(_df['games'])
).fit()

# Player-level aggregation to reduce repeated referee effects
agg = _df.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    redCards=('redCards', 'sum'),
    games=('games', 'sum')
).reset_index()
agg['red_per_game'] = agg['redCards'] / agg['games']
median_skin_player = agg['skin_tone'].median()
agg['dark'] = (agg['skin_tone'] > median_skin_player).astype(int)
rate_by_group_player = agg.groupby('dark')['red_per_game'].mean()

poisson_player = sm.GLM(
    agg['redCards'],
    sm.add_constant(agg['skin_tone']),
    family=sm.families.Poisson(),
    offset=np.log(agg['games'])
).fit()

# Report
print('Dyad-level median skin tone:', median_skin)
print('Dyad-level mean red per game (light vs dark):')
print(rate_by_group)
print('\nDyad-level Poisson regression (redCards ~ skin_tone, offset log(games))')
print(poisson_dyad.summary().tables[1])

print('\nPlayer-level mean red per game (light vs dark):')
print(rate_by_group_player)
print('\nPlayer-level Poisson regression (redCards ~ skin_tone, offset log(games))')
print(poisson_player.summary().tables[1])
