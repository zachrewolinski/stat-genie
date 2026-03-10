import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Skin tone: average of two raters when available
skin_cols = ['feature18', 'feature19']

df['skin_mean'] = df[skin_cols].mean(axis=1)

# Red cards count and games
red = df['feature16']
games = df['feature9']

# Filter rows with skin and games >0
analysis = df.loc[df['skin_mean'].notna() & games.notna() & (games > 0) & red.notna()].copy()

# Define light/dark categories for descriptive comparison
# Values are on 0,0.25,0.5,0.75,1.0 scale
analysis['skin_cat'] = pd.cut(
    analysis['skin_mean'],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=['light', 'medium', 'dark']
)

# Compute rates per game by category
analysis['red_rate'] = analysis['feature16'] / analysis['feature9']

rate_summary = analysis.groupby('skin_cat')['red_rate'].agg(['mean','median','count'])

# Poisson regression with offset for games
# Model: red_cards ~ skin_mean
X = sm.add_constant(analysis['skin_mean'])
poisson_model = sm.GLM(analysis['feature16'], X, family=sm.families.Poisson(), offset=np.log(analysis['feature9']))
poisson_res = poisson_model.fit(cov_type='HC3')

# Effect size
coef = poisson_res.params['skin_mean']
se = poisson_res.bse['skin_mean']
pval = poisson_res.pvalues['skin_mean']
rr = np.exp(coef)

# Predicted rates for light (0) vs dark (1)
# using mean offset = 1 game
rate_light = np.exp(poisson_res.params['const'] + coef*0)
rate_dark = np.exp(poisson_res.params['const'] + coef*1)

# Additional: simple comparison of dark vs light mean rate
light_rate = rate_summary.loc['light','mean'] if 'light' in rate_summary.index else np.nan
medium_rate = rate_summary.loc['medium','mean'] if 'medium' in rate_summary.index else np.nan
dark_rate = rate_summary.loc['dark','mean'] if 'dark' in rate_summary.index else np.nan

# Basic t-test (Welch) on red_rate between light and dark for context
light_rates = analysis.loc[analysis['skin_cat']=='light','red_rate']
dark_rates = analysis.loc[analysis['skin_cat']=='dark','red_rate']
from scipy import stats

ttest_p = np.nan
if len(light_rates) > 1 and len(dark_rates) > 1:
    ttest = stats.ttest_ind(light_rates, dark_rates, equal_var=False, nan_policy='omit')
    ttest_p = ttest.pvalue

print('Rows used:', len(analysis))
print('Rate summary by skin_cat:\n', rate_summary)
print('Poisson coef:', coef, 'se:', se, 'p:', pval, 'RR:', rr)
print('Predicted rate light (per game):', rate_light, 'dark:', rate_dark)
print('T-test p (red_rate light vs dark):', ttest_p)

# Save key outputs for later
out = {
    'rows_used': int(len(analysis)),
    'rate_summary': rate_summary.reset_index().to_dict(orient='records'),
    'coef': float(coef),
    'se': float(se),
    'pval': float(pval),
    'rr': float(rr),
    'rate_light': float(rate_light),
    'rate_dark': float(rate_dark),
    'ttest_p': float(ttest_p) if not np.isnan(ttest_p) else None,
}

with open('analysis_output.json','w') as f:
    json.dump(out,f,indent=2)
