import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute skin tone average
skin = df[['rater1','rater2']].mean(axis=1)
df = df.assign(skin_tone=skin)

# Drop rows with missing skin or games/redCards
analysis_df = df.dropna(subset=['skin_tone','games','redCards']).copy()
analysis_df = analysis_df[analysis_df['games'] > 0]

# Offset for exposure
analysis_df['log_games'] = np.log(analysis_df['games'])

# Poisson regression with robust SE
X_poisson = sm.add_constant(analysis_df['skin_tone'])
poisson_model = sm.GLM(analysis_df['redCards'], X_poisson,
                      family=sm.families.Poisson(), offset=analysis_df['log_games']).fit(cov_type='HC0')

# Logistic regression for any red card, with games as covariate
analysis_df['any_red'] = (analysis_df['redCards'] > 0).astype(int)
X_logit = sm.add_constant(analysis_df[['skin_tone','games']])
logit_model = sm.Logit(analysis_df['any_red'], X_logit).fit(disp=False, cov_type='HC0')

# Group comparison: extremes
light = analysis_df[analysis_df['skin_tone'] <= 0.25]
dark = analysis_df[analysis_df['skin_tone'] >= 0.75]

light_rate = light['redCards'].sum() / light['games'].sum() if light['games'].sum() > 0 else np.nan
dark_rate = dark['redCards'].sum() / dark['games'].sum() if dark['games'].sum() > 0 else np.nan
rate_ratio = dark_rate / light_rate if light_rate and not np.isnan(light_rate) else np.nan

# Output key stats
print('N total:', len(df))
print('N with skin tone:', len(analysis_df))

print('Poisson coef (skin_tone):', poisson_model.params['skin_tone'])
print('Poisson robust SE:', poisson_model.bse['skin_tone'])
print('Poisson p-value:', poisson_model.pvalues['skin_tone'])
print('Poisson IRR:', np.exp(poisson_model.params['skin_tone']))

print('Logit coef (skin_tone):', logit_model.params['skin_tone'])
print('Logit robust SE:', logit_model.bse['skin_tone'])
print('Logit p-value:', logit_model.pvalues['skin_tone'])
print('Logit OR:', np.exp(logit_model.params['skin_tone']))

print('Light rate:', light_rate)
print('Dark rate:', dark_rate)
print('Rate ratio (dark/light):', rate_ratio)

print('Skin tone unique values:', sorted(analysis_df['skin_tone'].dropna().unique()))

print('Red cards mean per dyad:', analysis_df['redCards'].mean())
print('Red cards per game overall:', analysis_df['redCards'].sum() / analysis_df['games'].sum())
