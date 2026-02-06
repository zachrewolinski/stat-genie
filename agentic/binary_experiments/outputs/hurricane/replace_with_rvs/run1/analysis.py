import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
_df = pd.read_csv('hurricane.csv')

# Outcomes and predictors
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Core model: log deaths vs femininity + intensity controls + year
formula = 'log_deaths ~ masfem + wind + min + year'
model = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Alternate model using binary name gender instead of masfem rating
formula_gender = 'log_deaths ~ gender_mf + wind + min + year'
model_gender = smf.ols(formula_gender, data=_df).fit(cov_type='HC3')

# Correlation for context
corr = _df[['masfem', 'log_deaths']].corr(method='spearman').iloc[0, 1]

# Print compact results
print('N:', len(_df))
print('Spearman corr(masfem, log_deaths):', round(corr, 3))
print('\nModel 1 (masfem):')
print(model.summary().tables[1])
print('\nModel 2 (gender_mf):')
print(model_gender.summary().tables[1])

# Save key results for downstream use
results = {
    'n': int(len(_df)),
    'spearman_masfem_log_deaths': float(corr),
    'masfem_coef': float(model.params['masfem']),
    'masfem_pvalue': float(model.pvalues['masfem']),
    'masfem_ci_low': float(model.conf_int().loc['masfem'][0]),
    'masfem_ci_high': float(model.conf_int().loc['masfem'][1]),
    'gender_mf_coef': float(model_gender.params['gender_mf']),
    'gender_mf_pvalue': float(model_gender.pvalues['gender_mf']),
    'gender_mf_ci_low': float(model_gender.conf_int().loc['gender_mf'][0]),
    'gender_mf_ci_high': float(model_gender.conf_int().loc['gender_mf'][1]),
}

pd.Series(results).to_json('analysis_results.json', orient='index')
