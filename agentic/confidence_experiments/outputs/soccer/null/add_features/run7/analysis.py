import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Select relevant columns
cols = ['playerShort', 'rater1', 'rater2', 'redCards', 'games']
for c in cols:
    if c not in _df.columns:
        raise ValueError(f'Missing column: {c}')

# Compute mean skin rating
skin_mean = _df[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Filter rows with skin info and valid games
_df = _df.copy()
_df['skin_mean'] = skin_mean
_df = _df[_df['skin_mean'].notna()]
_df = _df[_df['games'] > 0]

# Define skin categories
# 0.0, 0.25, 0.5, 0.75, 1.0 scale
_df['skin_cat'] = pd.cut(
    _df['skin_mean'],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=['light', 'medium', 'dark']
)

# Main binary comparison: light vs dark only
binary = _df[_df['skin_cat'].isin(['light', 'dark'])].copy()

# Aggregate rates by group
agg = binary.groupby('skin_cat').agg(
    redCards_total=('redCards', 'sum'),
    games_total=('games', 'sum'),
    dyads=('redCards', 'size'),
    players=('playerShort', 'nunique')
)
agg['rate_per_game'] = agg['redCards_total'] / agg['games_total']

# Rate ratio dark vs light
rate_ratio = (agg.loc['dark', 'rate_per_game'] / agg.loc['light', 'rate_per_game']) if 'dark' in agg.index and 'light' in agg.index else np.nan

# Poisson regression with offset(log(games))
# Predictor: dark indicator (1 if dark)
# Cluster-robust SE by playerShort
binary['dark'] = (binary['skin_cat'] == 'dark').astype(int)

y = binary['redCards']
X = sm.add_constant(binary['dark'])
offset = np.log(binary['games'])

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='cluster', cov_kwds={'groups': binary['playerShort']})

irr = np.exp(poisson_res.params['dark'])
ci = np.exp(poisson_res.conf_int().loc['dark'])
pval = poisson_res.pvalues['dark']

# Continuous model: per 0.25 increment in skin_mean
_df['skin_mean_025'] = _df['skin_mean'] / 0.25
Xc = sm.add_constant(_df['skin_mean_025'])
yc = _df['redCards']
offset_c = np.log(_df['games'])

poisson_model_c = sm.GLM(yc, Xc, family=sm.families.Poisson(), offset=offset_c)
poisson_res_c = poisson_model_c.fit(cov_type='cluster', cov_kwds={'groups': _df['playerShort']})

irr_c = np.exp(poisson_res_c.params['skin_mean_025'])
ci_c = np.exp(poisson_res_c.conf_int().loc['skin_mean_025'])
pval_c = poisson_res_c.pvalues['skin_mean_025']

# Simple rate comparison with chi-square (approx) using per-game counts as exposure
# (Not a standard chi-square; we will compute difference in proportions with z-test)
# Using aggregated counts of red cards and games (treating red cards as events).

# Print summary
print('Rows with skin info:', len(_df))
print('Binary rows:', len(binary))
print('\nAggregate (light vs dark):')
print(agg)
print('\nRate ratio (dark/light):', rate_ratio)
print('\nPoisson (dark vs light) IRR:', irr)
print('95% CI:', tuple(ci))
print('p-value:', pval)
print('\nPoisson continuous (per 0.25 skin increase) IRR:', irr_c)
print('95% CI:', tuple(ci_c))
print('p-value:', pval_c)
