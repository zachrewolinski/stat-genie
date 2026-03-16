import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
df = pd.read_csv('hurricane.csv')

# Basic transforms
df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam'] = np.log1p(df['ndam'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Simple correlations
corrs = {
    'masfem_vs_deaths': df['masfem'].corr(df['alldeaths']),
    'masfem_vs_log_deaths': df['masfem'].corr(df['log_deaths']),
    'masfem_mturk_vs_deaths': df['masfem_mturk'].corr(df['alldeaths']),
    'gender_mf_vs_deaths': df['gender_mf'].corr(df['alldeaths']),
}

# OLS models: log deaths ~ masfem + controls
# Controls for intensity: wind, min pressure, category, damage
# We'll use log_ndam as proxy for exposure, or include ndam as control

models = {}

formula1 = 'log_deaths ~ masfem'
formula2 = 'log_deaths ~ masfem + wind + min + category'
formula3 = 'log_deaths ~ masfem + wind + min + category + log_ndam'
formula4 = 'log_deaths ~ masfem + wind + min + category + log_ndam15'

for i, f in enumerate([formula1, formula2, formula3, formula4], start=1):
    model = smf.ols(f, data=df).fit(cov_type='HC3')
    models[f'ols_{i}'] = model

# GLM Poisson with log link for counts
# Use exposure? We'll not set offset; include controls similarly
poisson_models = {}
for i, f in enumerate([
    'alldeaths ~ masfem',
    'alldeaths ~ masfem + wind + min + category',
    'alldeaths ~ masfem + wind + min + category + log_ndam',
    'alldeaths ~ masfem + wind + min + category + log_ndam15'
], start=1):
    model = smf.glm(f, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
    poisson_models[f'poisson_{i}'] = model

# Negative binomial
nb_models = {}
for i, f in enumerate([
    'alldeaths ~ masfem',
    'alldeaths ~ masfem + wind + min + category',
    'alldeaths ~ masfem + wind + min + category + log_ndam',
    'alldeaths ~ masfem + wind + min + category + log_ndam15'
], start=1):
    model = smf.glm(f, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    nb_models[f'nb_{i}'] = model

# Print key results
print('Correlations:')
for k, v in corrs.items():
    print(k, v)

print('\nOLS models (log deaths):')
for name, m in models.items():
    coef = m.params['masfem']
    pval = m.pvalues['masfem']
    print(name, 'coef', coef, 'p', pval, 'n', int(m.nobs), 'r2', m.rsquared)

print('\nPoisson models (deaths):')
for name, m in poisson_models.items():
    coef = m.params['masfem']
    pval = m.pvalues['masfem']
    print(name, 'coef', coef, 'p', pval, 'n', int(m.nobs), 'pseudo_r2', 1 - m.deviance/m.null_deviance)

print('\nNegBin models (deaths):')
for name, m in nb_models.items():
    coef = m.params['masfem']
    pval = m.pvalues['masfem']
    print(name, 'coef', coef, 'p', pval, 'n', int(m.nobs))

# Also try masfem_mturk for robustness in OLS with controls
mturk_model = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + log_ndam', data=df).fit(cov_type='HC3')
print('\nMTurk masfem model: coef', mturk_model.params['masfem_mturk'], 'p', mturk_model.pvalues['masfem_mturk'])

# Gender indicator model
gender_model = smf.ols('log_deaths ~ gender_mf + wind + min + category + log_ndam', data=df).fit(cov_type='HC3')
print('Gender model: coef', gender_model.params['gender_mf'], 'p', gender_model.pvalues['gender_mf'])

# Extra diagnostics: correlation with severity and year, and OLS with year controls
print('\nCorrelations with severity/year:')
for col in ['wind', 'min', 'category', 'year']:
    print('masfem_vs_' + col, df['masfem'].corr(df[col]))

# OLS with year control
ols_year = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')
print('\nOLS with year control: coef', ols_year.params['masfem'], 'p', ols_year.pvalues['masfem'])

# Discrete Negative Binomial with estimated alpha
# Use same controls; drop rows with missing in log_ndam
nb_df = df[['alldeaths', 'masfem', 'wind', 'min', 'category', 'log_ndam']].dropna()
X = sm.add_constant(nb_df[['masfem', 'wind', 'min', 'category', 'log_ndam']])
nb_model = NegativeBinomial(nb_df['alldeaths'], X).fit(disp=False)
print('\nDiscrete NB (estimated alpha) with log_ndam: coef', nb_model.params['masfem'], 'p', nb_model.pvalues['masfem'], 'alpha', nb_model.params['alpha'])
