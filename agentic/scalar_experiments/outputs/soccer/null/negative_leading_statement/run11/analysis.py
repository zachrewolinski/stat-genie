import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'soccer.csv'

df = pd.read_csv(path)

# Create mean skin tone (0-1 scale) for players with photos
# Use average of rater1 and rater2, ignoring missing
skin = df[['rater1','rater2']].mean(axis=1)

df = df.copy()

df['skin_mean'] = skin

# Basic counts
print('rows', len(df))
print('missing skin', df['skin_mean'].isna().mean())

# Define dark vs light based on quartiles? We'll use >=0.5 as darker (since 0-1 scale 5-point normalized: 0,0.25,0.5,0.75,1)
# So 0-0.25 light, 0.75-1 dark? But to compare dark vs light, we can use extremes to avoid ambiguous middle.
# We'll create two categories: light (<=0.25) and dark (>=0.75). Also compute 0.5 threshold.

df['light'] = df['skin_mean'] <= 0.25
df['dark'] = df['skin_mean'] >= 0.75

print('light share', df['light'].mean())
print('dark share', df['dark'].mean())

# Aggregate by player to avoid repeated dyads? But question about red cards from referees; dyad-level appropriate with exposure.
# We'll model red cards count with Poisson with log(games) offset.

# Keep rows with skin info and games>0
model_df = df[df['skin_mean'].notna() & (df['games']>0)].copy()

# Poisson regression: redCards ~ skin_mean + games offset; maybe control for position, league, etc.
# But minimal to answer question. We'll run:
# 1) Poisson with offset log(games)
# 2) Logistic for any red card

model_df['any_red'] = (model_df['redCards']>0).astype(int)
model_df['log_games'] = np.log(model_df['games'])

# Poisson
poisson = smf.glm('redCards ~ skin_mean', data=model_df, family=sm.families.Poisson(), offset=model_df['log_games']).fit()
print(poisson.summary())

# Logistic
logit = smf.glm('any_red ~ skin_mean', data=model_df, family=sm.families.Binomial()).fit()
print(logit.summary())

# Compare means for light vs dark (extremes)
light_df = model_df[model_df['light']]
dark_df = model_df[model_df['dark']]

# rate per game
light_rate = (light_df['redCards'].sum() / light_df['games'].sum()) if len(light_df)>0 else np.nan

dark_rate = (dark_df['redCards'].sum() / dark_df['games'].sum()) if len(dark_df)>0 else np.nan

print('light rate per game', light_rate)
print('dark rate per game', dark_rate)

# Poisson on extremes with indicator
ext_df = model_df[model_df['light'] | model_df['dark']].copy()
ext_df['dark_group'] = ext_df['dark'].astype(int)
poisson_ext = smf.glm('redCards ~ dark_group', data=ext_df, family=sm.families.Poisson(), offset=np.log(ext_df['games'])).fit()
print(poisson_ext.summary())

# print counts
print('ext rows', len(ext_df))
print('ext dark count', ext_df['dark_group'].sum())

# store key stats
import json

results = {
    'poisson_coef': poisson.params['skin_mean'],
    'poisson_p': poisson.pvalues['skin_mean'],
    'poisson_ci': poisson.conf_int().loc['skin_mean'].tolist(),
    'logit_coef': logit.params['skin_mean'],
    'logit_p': logit.pvalues['skin_mean'],
    'logit_ci': logit.conf_int().loc['skin_mean'].tolist(),
    'light_rate_per_game': light_rate,
    'dark_rate_per_game': dark_rate,
    'poisson_ext_coef': poisson_ext.params['dark_group'],
    'poisson_ext_p': poisson_ext.pvalues['dark_group'],
    'poisson_ext_ci': poisson_ext.conf_int().loc['dark_group'].tolist(),
    'n_rows': len(model_df),
    'n_ext': len(ext_df)
}

print(json.dumps(results, indent=2))

