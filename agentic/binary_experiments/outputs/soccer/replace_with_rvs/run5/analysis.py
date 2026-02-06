import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Compute average skin tone from raters
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)

df = df.assign(skin_tone=skin)

# Define light vs dark, excluding neutral (0.5) and missing
# 0=very light, 0.25=light, 0.5=neutral, 0.75=dark, 1=very dark
mask = df['skin_tone'].notna()
light = mask & (df['skin_tone'] < 0.5)
dark = mask & (df['skin_tone'] > 0.5)

subset = df[light | dark].copy()
subset['dark'] = (subset['skin_tone'] > 0.5).astype(int)

# Aggregate red card rates per game
agg = subset.groupby('dark').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    dyads=('redCards', 'size')
).reset_index()
agg['rate_per_game'] = agg['red_cards'] / agg['games']

# Poisson regression with offset log(games)
# Model redCards ~ dark with exposure games
subset = subset[subset['games'] > 0].copy()
subset['log_games'] = np.log(subset['games'])

X = sm.add_constant(subset['dark'])
model = sm.GLM(subset['redCards'], X, family=sm.families.Poisson(), offset=subset['log_games'])
res = model.fit(cov_type='HC1')

# Extract rate ratio for dark vs light
coef = res.params['dark']
se = res.bse['dark']
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

print('Group summary (dark=0 light, dark=1 dark):')
print(agg)
print('\nPoisson regression with offset log(games):')
print(res.summary().tables[1])
print(f"\nRate ratio (dark vs light): {rr:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f})")
