import pandas as pd
import numpy as np
import statsmodels.api as sm

csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Ensure numeric columns
for col in ['rater1', 'rater2', 'redCards', 'games']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute mean skin tone (0 to 1 scale)
df['skin_mean'] = df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with needed data
analysis_df = df[['skin_mean', 'redCards', 'games']].dropna()
analysis_df = analysis_df[(analysis_df['games'] > 0)]

# Basic counts
n_total = len(analysis_df)

# Define light and dark categories on 5-point scale
# rater values are normalized: 0.0, 0.25, 0.5, 0.75, 1.0
light_mask = analysis_df['skin_mean'] <= 0.25
dark_mask = analysis_df['skin_mean'] >= 0.75

light_df = analysis_df[light_mask]
dark_df = analysis_df[dark_mask]

# Rates per game
light_rate = (light_df['redCards'].sum() / light_df['games'].sum()) if light_df['games'].sum() > 0 else np.nan
dark_rate = (dark_df['redCards'].sum() / dark_df['games'].sum()) if dark_df['games'].sum() > 0 else np.nan

# Poisson regression on all data (continuous skin tone)
# Model redCards with offset log(games)
analysis_df['log_games'] = np.log(analysis_df['games'])
X = sm.add_constant(analysis_df['skin_mean'])
poisson_model = sm.GLM(analysis_df['redCards'], X, family=sm.families.Poisson(), offset=analysis_df['log_games']).fit(cov_type='HC0')

coef = poisson_model.params['skin_mean']
se = poisson_model.bse['skin_mean']
pval = poisson_model.pvalues['skin_mean']
irr = np.exp(coef)

# Poisson regression for dark vs light only
subset = analysis_df[light_mask | dark_mask].copy()
subset['dark'] = (subset['skin_mean'] >= 0.75).astype(int)
subset['log_games'] = np.log(subset['games'])
X2 = sm.add_constant(subset['dark'])
poisson_model_bin = sm.GLM(subset['redCards'], X2, family=sm.families.Poisson(), offset=subset['log_games']).fit(cov_type='HC0')
coef_bin = poisson_model_bin.params['dark']
se_bin = poisson_model_bin.bse['dark']
pval_bin = poisson_model_bin.pvalues['dark']
irr_bin = np.exp(coef_bin)

# Summary output
print('Total rows used:', n_total)
print('Light rows:', len(light_df), 'Dark rows:', len(dark_df))
print('Light red card rate per game:', light_rate)
print('Dark red card rate per game:', dark_rate)
print('Poisson (continuous skin_mean) coef:', coef, 'SE:', se, 'p:', pval, 'IRR:', irr)
print('Poisson (dark vs light) coef:', coef_bin, 'SE:', se_bin, 'p:', pval_bin, 'IRR:', irr_bin)

# Extra: check red cards prevalence
print('Overall red cards per game:', analysis_df['redCards'].sum() / analysis_df['games'].sum())
print('Any red card rate (rows with redCards>0):', (analysis_df['redCards'] > 0).mean())
