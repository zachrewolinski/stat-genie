import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest

# Load data
path = 'soccer.csv'

df = pd.read_csv(path)

# Skin tone average
skin_mean = df[['rater1','rater2']].mean(axis=1)

df = df.assign(skin_mean=skin_mean)

# Exclude missing skin or games
base = df.dropna(subset=['skin_mean','games','redCards']).copy()

# Poisson regression with offset log(games)
base['log_games'] = np.log(base['games'])

X = sm.add_constant(base['skin_mean'])
model = sm.GLM(base['redCards'], X, family=sm.families.Poisson(), offset=base['log_games'])
res = model.fit(cov_type='HC1')

print('Poisson GLM with skin_mean (0-1):')
print(res.summary())

# Incidence rate ratio per 1-unit increase in skin_mean
coef = res.params['skin_mean']
se = res.bse['skin_mean']
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('IRR per 1.0 skin_mean increase:', irr, '95% CI', ci_low, ci_high, 'p', res.pvalues['skin_mean'])

# Compare dark vs light groups
light = base[base['skin_mean'] <= 0.25]
dark = base[base['skin_mean'] >= 0.75]

# Poisson rate ratio using GLM with indicator
sub = pd.concat([light.assign(group=0), dark.assign(group=1)], ignore_index=True)
sub['log_games'] = np.log(sub['games'])
X2 = sm.add_constant(sub['group'])
model2 = sm.GLM(sub['redCards'], X2, family=sm.families.Poisson(), offset=sub['log_games'])
res2 = model2.fit(cov_type='HC1')

coef2 = res2.params['group']
se2 = res2.bse['group']
irr2 = np.exp(coef2)
ci2_low = np.exp(coef2 - 1.96*se2)
ci2_high = np.exp(coef2 + 1.96*se2)

print('\nDark (>=0.75) vs Light (<=0.25) Poisson GLM:')
print(res2.summary())
print('IRR dark vs light:', irr2, '95% CI', ci2_low, ci2_high, 'p', res2.pvalues['group'])

# Basic rate comparison
rate_light = light['redCards'].sum() / light['games'].sum()
rate_dark = dark['redCards'].sum() / dark['games'].sum()
print('\nRates per game: light', rate_light, 'dark', rate_dark)
print('Counts: light redCards', light['redCards'].sum(), 'games', light['games'].sum(), 'n dyads', len(light))
print('Counts: dark redCards', dark['redCards'].sum(), 'games', dark['games'].sum(), 'n dyads', len(dark))

# Logistic regression on any red card with games covariate and skin_mean
base['any_red'] = (base['redCards'] > 0).astype(int)
X3 = sm.add_constant(base[['skin_mean','games']])
logit = sm.Logit(base['any_red'], X3)
res3 = logit.fit(disp=False, maxiter=200)
print('\nLogistic regression any_red ~ skin_mean + games:')
print(res3.summary())

# odds ratio for skin_mean
coef3 = res3.params['skin_mean']
se3 = res3.bse['skin_mean']
or3 = np.exp(coef3)
ci3_low = np.exp(coef3 - 1.96*se3)
ci3_high = np.exp(coef3 + 1.96*se3)
print('OR per 1.0 skin_mean increase:', or3, '95% CI', ci3_low, ci3_high, 'p', res3.pvalues['skin_mean'])

