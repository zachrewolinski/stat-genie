import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Map columns based on metadata descriptions
# games (number of games in player-referee dyad) is in column named 'redCards'
# red card count is in column named 'yellowCards'
# skin tone ratings in columns 'rater1' and 'nExp'

# Build skin tone mean
skin = df[['rater1','nExp']].mean(axis=1, skipna=True)

df = df.assign(skin_mean=skin)

# Drop rows without skin ratings or with missing games/red cards
analysis = df.dropna(subset=['skin_mean','redCards','yellowCards']).copy()

# Ensure games positive
analysis = analysis[analysis['redCards'] > 0]

# Define dark vs light using midpoint threshold
analysis['dark'] = (analysis['skin_mean'] >= 0.5).astype(int)

# Aggregate rates
agg = analysis.groupby('dark').agg(
    red_cards=('yellowCards','sum'),
    games=('redCards','sum'),
    dyads=('yellowCards','size'),
    mean_skin=('skin_mean','mean')
).reset_index()
agg['rate_per_game'] = agg['red_cards'] / agg['games']

print('Group summary')
print(agg)

# Poisson regression with offset log(games)
analysis['log_games'] = np.log(analysis['redCards'])

# Use robust SE (HC1)
model = smf.glm('yellowCards ~ dark', data=analysis,
                family=sm.families.Poisson(), offset=analysis['log_games']).fit(cov_type='HC1')

print('\nPoisson dark vs light')
print(model.summary())

coef = model.params['dark']
se = model.bse['dark']
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

print(f'Rate ratio dark vs light: {rr:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f}), p={model.pvalues["dark"]:.4g}')

# Also continuous skin tone model
model_cont = smf.glm('yellowCards ~ skin_mean', data=analysis,
                     family=sm.families.Poisson(), offset=analysis['log_games']).fit(cov_type='HC1')
coef_c = model_cont.params['skin_mean']
se_c = model_cont.bse['skin_mean']
rr_c = np.exp(coef_c)
ci_low_c = np.exp(coef_c - 1.96*se_c)
ci_high_c = np.exp(coef_c + 1.96*se_c)
print('\nPoisson continuous skin tone')
print(f'Rate ratio per 1.0 skin_mean: {rr_c:.3f} (95% CI {ci_low_c:.3f}, {ci_high_c:.3f}), p={model_cont.pvalues["skin_mean"]:.4g}')

# Simple rate comparison test (two-proportion style using poisson approximation)
# compute log rate ratio and its SE
rate_dark = agg.loc[agg['dark']==1, 'red_cards'].iloc[0] / agg.loc[agg['dark']==1, 'games'].iloc[0] if (agg['dark']==1).any() else np.nan
rate_light = agg.loc[agg['dark']==0, 'red_cards'].iloc[0] / agg.loc[agg['dark']==0, 'games'].iloc[0] if (agg['dark']==0).any() else np.nan
print(f'Rate dark: {rate_dark:.6f} per game, light: {rate_light:.6f} per game')
