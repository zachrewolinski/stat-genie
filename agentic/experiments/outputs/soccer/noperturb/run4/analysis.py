import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Skin tone: mean of two raters
df['skin_mean'] = df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with skin ratings and games > 0
df = df[df['skin_mean'].notna() & (df['games'] > 0)].copy()

# Define dark vs light (dark > 0.5 on 0-1 scale)
df['dark'] = (df['skin_mean'] > 0.5).astype(int)

# Descriptive rates
df['red_per_game'] = df['redCards'] / df['games']

summary = df.groupby('dark').agg(
    n=('dark', 'size'),
    total_games=('games', 'sum'),
    total_red=('redCards', 'sum'),
    mean_red_per_game=('red_per_game', 'mean'),
    any_red_rate=('redCards', lambda x: (x > 0).mean())
).reset_index()

# Poisson regression with offset for games
# Model: redCards ~ dark, offset log(games)
X = sm.add_constant(df['dark'])
offset = np.log(df['games'])
model = sm.GLM(df['redCards'], X, family=sm.families.Poisson(), offset=offset)
result = model.fit(cov_type='HC0')

# Extract effect
coef_dark = result.params['dark']
rr_dark = np.exp(coef_dark)
ci_low, ci_high = np.exp(result.conf_int().loc['dark'])

# Save key outputs for inspection
print('Summary by dark:')
print(summary)
print('\nPoisson GLM (robust SE)')
print(result.summary().as_text())
print('\nRate ratio for dark vs light:', rr_dark)
print('95% CI:', (ci_low, ci_high))
