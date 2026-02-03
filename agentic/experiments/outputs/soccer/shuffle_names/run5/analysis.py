import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Identify columns based on distributions and dataset structure
# Games per player-referee dyad: integer count with min>=1 and max in ~[10, 60]
# In this dataset, that corresponds to column named 'redCards'.
games_col = 'redCards'

# Red cards per dyad: very low count, max around 2
# In this dataset, that corresponds to column named 'yellowCards'.
red_cards_col = 'yellowCards'

# Skin tone ratings: rater1 and nExp (both 0-1)
skin_cols = ['rater1', 'nExp']

# Drop rows with missing skin ratings
mask = df[skin_cols].notna().all(axis=1)
sub = df.loc[mask].copy()

# Mean skin tone on 0-1 scale; higher = darker
sub['skin_mean'] = sub[skin_cols].mean(axis=1)

# Binary dark vs light
sub['dark'] = (sub['skin_mean'] >= 0.5).astype(int)

# Compute red card rates per game by group
rate_dark = sub.loc[sub['dark'] == 1, red_cards_col].sum() / sub.loc[sub['dark'] == 1, games_col].sum()
rate_light = sub.loc[sub['dark'] == 0, red_cards_col].sum() / sub.loc[sub['dark'] == 0, games_col].sum()
rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan

print('Rows with skin ratings:', len(sub))
print('Red card rate per game (dark):', rate_dark)
print('Red card rate per game (light):', rate_light)
print('Rate ratio (dark / light):', rate_ratio)

# Poisson regression with offset log(games) to test association
# red_cards ~ dark, offset = log(games)
X = sm.add_constant(sub['dark'])
model = sm.GLM(sub[red_cards_col], X, family=sm.families.Poisson(), offset=np.log(sub[games_col]))
res = model.fit()

coef = res.params['dark']
rr = np.exp(coef)
ci = res.conf_int().loc['dark'].values
ci_rr = np.exp(ci)

print('\nPoisson regression (red_cards ~ dark + offset log(games))')
print('Rate ratio (dark vs light):', rr)
print('95% CI:', ci_rr)
print('p-value:', res.pvalues['dark'])
