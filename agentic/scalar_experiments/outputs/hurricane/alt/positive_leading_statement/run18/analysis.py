import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from pathlib import Path

DATA_PATH = Path('hurricane.csv')

df = pd.read_csv(DATA_PATH)

# Basic checks
print('rows', len(df))
deaths_mean = df['alldeaths'].mean()
deaths_var = df['alldeaths'].var()
zero_deaths = int((df['alldeaths'] == 0).sum())
print('alldeaths mean', deaths_mean, 'var', deaths_var, 'zeros', zero_deaths)

# Outcome: deaths. Use log1p to handle zeros and skew.
df['log_deaths'] = np.log1p(df['alldeaths'])

# Key predictor: masfem (higher = more feminine)

# Control variables: wind (intensity), min pressure (lower means stronger), category.
# Avoid multicollinearity by selecting a smaller, standard set; use wind and min and category.

X = df[['masfem', 'wind', 'min', 'category']].copy()
X = sm.add_constant(X)
model = sm.OLS(df['log_deaths'], X).fit()
print(model.summary())

# Alternative: using masfem_mturk
X2 = df[['masfem_mturk', 'wind', 'min', 'category']].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['log_deaths'], X2).fit()
print('\nMTurk masfem model')
print(model2.summary())

# Simple correlation
corr = df['masfem'].corr(df['log_deaths'])
print('\nCorrelation masfem vs log_deaths:', corr)

# Median split for descriptive difference
median_masfem = df['masfem'].median()
low = df[df['masfem'] <= median_masfem]['log_deaths']
high = df[df['masfem'] > median_masfem]['log_deaths']
print('Median masfem:', median_masfem)
print('Mean log_deaths low:', low.mean(), 'high:', high.mean())

# Robust regression as sensitivity
rmodel = sm.RLM(df['log_deaths'], X, M=sm.robust.norms.HuberT()).fit()
print('\nRobust (Huber) model')
print(rmodel.summary())

# Poisson for counts (with overdispersion) using GLM Poisson with log link
# Use alldeaths + 1 to avoid zeros? Poisson can handle zeros; use count as-is.
glm = sm.GLM(df['alldeaths'], X, family=sm.families.Poisson()).fit()
print('\nPoisson GLM')
print(glm.summary())

# Negative binomial GLM to account for overdispersion
nb_glm = sm.GLM(df['alldeaths'], X, family=sm.families.NegativeBinomial()).fit()
print('\nNegative Binomial GLM')
print(nb_glm.summary())

# Negative binomial (NB2) with estimated dispersion
nb2 = smd.NegativeBinomial(df['alldeaths'], X).fit(disp=False)
print('\nNegative Binomial (NB2) discrete model')
print(nb2.summary())

# Store key results for convenience
results = {
    'ols_masfem_coef': model.params['masfem'],
    'ols_masfem_p': model.pvalues['masfem'],
    'ols_masfem_ci': model.conf_int().loc['masfem'].tolist(),
    'ols_r2': model.rsquared,
    'ols_mturk_coef': model2.params['masfem_mturk'],
    'ols_mturk_p': model2.pvalues['masfem_mturk'],
    'ols_mturk_ci': model2.conf_int().loc['masfem_mturk'].tolist(),
    'robust_masfem_coef': rmodel.params['masfem'],
    'robust_masfem_p': rmodel.pvalues['masfem'],
    'poisson_masfem_coef': glm.params['masfem'],
    'poisson_masfem_p': glm.pvalues['masfem'],
    'nb_masfem_coef': nb_glm.params['masfem'],
    'nb_masfem_p': nb_glm.pvalues['masfem'],
    'nb2_masfem_coef': nb2.params['masfem'],
    'nb2_masfem_p': nb2.pvalues['masfem'],
    'nb2_alpha': float(nb2.params.get('alpha', float('nan'))),
    'deaths_mean': float(deaths_mean),
    'deaths_var': float(deaths_var),
    'zero_deaths': zero_deaths,
    'corr': corr,
    'median_masfem': float(median_masfem),
    'mean_log_deaths_low': float(low.mean()),
    'mean_log_deaths_high': float(high.mean()),
}

print('\nKey results JSON')
print(json.dumps(results, indent=2))
