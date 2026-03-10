import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = 'soccer.csv'
df = pd.read_csv(path)

# Basic checks
print('rows', len(df), 'cols', df.shape[1])
print('columns', list(df.columns))

# Construct skin tone measure if available
# Use average of rater1 and rater2; drop rows missing both
r1 = df['rater1'] if 'rater1' in df.columns else None
r2 = df['rater2'] if 'rater2' in df.columns else None

if r1 is not None and r2 is not None:
    df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)
    # keep rows with at least one rating
    df = df[df['skin_tone'].notna()].copy()
else:
    raise SystemExit('Skin tone ratings not found')

# Red cards and games
if 'redCards' not in df.columns:
    raise SystemExit('redCards missing')
if 'games' not in df.columns:
    raise SystemExit('games missing')

# For rate calculations, ensure games > 0
# Remove rows with non-positive games
rate_df = df[df['games'] > 0].copy()

print('rows with skin tone ratings:', len(df))
print('rows with games>0:', len(rate_df))

# Create binary light vs dark groups based on skin tone scale (0-1 with 5 points)
# Define light <= 0.25, dark >= 0.75 for a clean contrast
rate_df['skin_group'] = np.where(rate_df['skin_tone'] <= 0.25, 'light',
                         np.where(rate_df['skin_tone'] >= 0.75, 'dark', 'mid'))

# Summary by group
summary = rate_df.groupby('skin_group').agg(
    dyads=('redCards', 'size'),
    players=('playerShort', 'nunique'),
    total_games=('games', 'sum'),
    total_red=('redCards', 'sum'),
    mean_red=('redCards', 'mean'),
    mean_games=('games', 'mean'),
)
summary['red_per_game'] = summary['total_red'] / summary['total_games']
print('\nGroup summary (light/mid/dark):')
print(summary)

# Compare light vs dark using rate per game and Poisson regression with offset log(games)
ld = rate_df[rate_df['skin_group'].isin(['light', 'dark'])].copy()
print('\nLight/Dark counts:', ld['skin_group'].value_counts())

# Poisson regression: redCards ~ skin_tone (continuous) with offset log(games)
# Use all rows with ratings
# Add small constant to games to avoid log(0) though already filtered
poisson_df = rate_df.copy()
poisson_df['log_games'] = np.log(poisson_df['games'])

X = sm.add_constant(poisson_df['skin_tone'])
y = poisson_df['redCards']
poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=poisson_df['log_games'])
poisson_res = poisson_model.fit(cov_type='HC0')
print('\nPoisson regression (redCards ~ skin_tone, offset log(games))')
print(poisson_res.summary())

# Negative binomial as robustness
try:
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=poisson_df['log_games'])
    nb_res = nb_model.fit(cov_type='HC0')
    print('\nNegative Binomial regression (alpha=1.0)')
    print(nb_res.summary())
except Exception as e:
    print('NB regression failed:', e)

# Logistic regression on any red card in dyad
poisson_df['any_red'] = (poisson_df['redCards'] > 0).astype(int)
logit_model = sm.Logit(poisson_df['any_red'], X)
logit_res = logit_model.fit(disp=False)
print('\nLogistic regression (any red card ~ skin_tone)')
print(logit_res.summary())

# Effect size: difference in red cards per game for light vs dark
ld_summary = ld.groupby('skin_group').agg(total_red=('redCards','sum'), total_games=('games','sum'))
ld_summary['red_per_game'] = ld_summary['total_red'] / ld_summary['total_games']
print('\nLight vs Dark red per game:')
print(ld_summary)

# Additional: linear regression on red per game at dyad level (weighted by games)
poisson_df['red_per_game'] = poisson_df['redCards'] / poisson_df['games']
wls_model = sm.WLS(poisson_df['red_per_game'], X, weights=poisson_df['games'])
wls_res = wls_model.fit(cov_type='HC0')
print('\nWLS regression (red per game ~ skin_tone, weights=games)')
print(wls_res.summary())

# Output key metrics for later use
print('\nKey metrics:')
print('Poisson skin_tone coef:', poisson_res.params['skin_tone'], 'p=', poisson_res.pvalues['skin_tone'])
print('Logit skin_tone coef:', logit_res.params['skin_tone'], 'p=', logit_res.pvalues['skin_tone'])
print('WLS skin_tone coef:', wls_res.params['skin_tone'], 'p=', wls_res.pvalues['skin_tone'])
print('Light red per game:', ld_summary.loc['light','red_per_game'] if 'light' in ld_summary.index else None)
print('Dark red per game:', ld_summary.loc['dark','red_per_game'] if 'dark' in ld_summary.index else None)
