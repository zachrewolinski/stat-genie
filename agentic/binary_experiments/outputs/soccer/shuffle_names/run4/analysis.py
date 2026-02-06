import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Column mapping based on metadata and value ranges:
# - rater1 and nExp are the two skin tone ratings (0-1 normalized)
# - yellowCards is the red-card count (0-2)
# - redCards is the number of games in the dyad (1-47)
skin = _df[['rater1', 'nExp']].mean(axis=1)
red_cards = _df['yellowCards']
games = _df['redCards']

# Keep rows with skin ratings and valid exposure
mask = skin.notna() & red_cards.notna() & games.notna() & (games > 0)
sub = _df.loc[mask].copy()
sub['skin'] = skin[mask]
sub['red_cards'] = red_cards[mask]
sub['games'] = games[mask]

# Simple rate comparison: dark (>0.5) vs light (<=0.5)
sub['group'] = np.where(sub['skin'] > 0.5, 'dark', 'light')
rate_by_group = sub.groupby('group').apply(lambda g: g['red_cards'].sum() / g['games'].sum())

# Poisson regression for red-card rate with exposure adjustment
X = sm.add_constant(sub['skin'])
poisson = sm.GLM(sub['red_cards'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
poisson_res = poisson.fit()

# Output key results
print('Red-card rate per game (dark vs light):')
print(rate_by_group)
print('Rate ratio (dark/light):', float(rate_by_group['dark'] / rate_by_group['light']))
print('\nPoisson regression coefficient for skin tone:')
print(poisson_res.params)
print('p-value for skin tone:', float(poisson_res.pvalues['skin']))
