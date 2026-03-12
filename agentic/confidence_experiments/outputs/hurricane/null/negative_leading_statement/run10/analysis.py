import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('hurricane.csv')

# Create log1p deaths

df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic correlations

corr_pearson = df['masfem'].corr(df['alldeaths'])
corr_spearman = df['masfem'].corr(df['alldeaths'], method='spearman')

# Spearman with log deaths
corr_spearman_log = df['masfem'].corr(df['log_deaths'], method='spearman')

# Simple OLS
model_simple = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Controls for intensity (wind, min pressure, category) and year
model_controls = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')

# Alternative rating (masfem_mturk)
model_controls_mturk = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=df).fit(cov_type='HC3')

# Binary gender indicator
model_controls_gender = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit(cov_type='HC3')

# Negative binomial regression on deaths (counts)
# Add small constant to avoid issues? no, NB handles zeros
nb_model = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df,
                   family=sm.families.NegativeBinomial()).fit()

results = {
    'pearson_corr_masfem_alldeaths': corr_pearson,
    'spearman_corr_masfem_alldeaths': corr_spearman,
    'spearman_corr_masfem_log_deaths': corr_spearman_log,
    'simple_ols_coef': model_simple.params['masfem'],
    'simple_ols_p': model_simple.pvalues['masfem'],
    'controls_ols_coef': model_controls.params['masfem'],
    'controls_ols_p': model_controls.pvalues['masfem'],
    'controls_mturk_coef': model_controls_mturk.params['masfem_mturk'],
    'controls_mturk_p': model_controls_mturk.pvalues['masfem_mturk'],
    'controls_gender_coef': model_controls_gender.params['gender_mf'],
    'controls_gender_p': model_controls_gender.pvalues['gender_mf'],
    'nb_coef': nb_model.params['masfem'],
    'nb_p': nb_model.pvalues['masfem']
}

print(results)

print('\nModel summaries (short):')
print(model_simple.summary().tables[1])
print(model_controls.summary().tables[1])
print(model_controls_mturk.summary().tables[1])
print(model_controls_gender.summary().tables[1])
print(nb_model.summary().tables[1])
