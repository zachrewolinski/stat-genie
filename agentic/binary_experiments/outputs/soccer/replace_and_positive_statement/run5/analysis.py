import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Compute mean skin tone from raters (0=very light, 1=very dark)
# Use mean of available ratings; drop rows with both missing
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
df = df.assign(mean_skin=skin)
df = df[~df['mean_skin'].isna()].copy()

# Define dark vs light using midpoint (>=0.5 is darker half of scale)
df['dark_skin'] = (df['mean_skin'] >= 0.5).astype(int)

# Basic group rates
df['red_per_game'] = df['redCards'] / df['games']
group_summary = df.groupby('dark_skin').agg(
    n=('redCards', 'size'),
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum'),
    mean_red_per_game=('red_per_game', 'mean'),
)
group_summary['red_rate_per_game'] = group_summary['total_red'] / group_summary['total_games']

# Poisson regression with exposure offset (games)
# Use log(games) as offset for exposure
model_df = df[['redCards', 'dark_skin', 'games']].copy()
model_df = model_df[model_df['games'] > 0]

X = sm.add_constant(model_df['dark_skin'])
y = model_df['redCards']
offset = np.log(model_df['games'])

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Rate ratio for dark skin
coef = poisson_res.params['dark_skin']
rr = np.exp(coef)
pval = poisson_res.pvalues['dark_skin']

print('Group summary (dark_skin=0 light, 1 dark):')
print(group_summary)
print('\nPoisson regression with exposure offset:')
print(poisson_res.summary())
print(f"\nRate ratio (dark vs light): {rr:.3f}, p-value: {pval:.4g}")
