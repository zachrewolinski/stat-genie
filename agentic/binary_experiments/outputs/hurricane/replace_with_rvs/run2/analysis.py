import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('hurricane.csv')

# Basic derived variables
df['log_deaths'] = np.log(df['alldeaths'])

# Controls commonly used for storm intensity
controls = ['wind', 'min', 'category', 'ndam15', 'year']

# OLS on log deaths
formula_masfem = 'log_deaths ~ masfem + ' + ' + '.join(controls)
formula_gender = 'log_deaths ~ gender_mf + ' + ' + '.join(controls)
formula_mturk = 'log_deaths ~ masfem_mturk + ' + ' + '.join(controls)

ols_masfem = smf.ols(formula_masfem, data=df).fit(cov_type='HC3')
ols_gender = smf.ols(formula_gender, data=df).fit(cov_type='HC3')
ols_mturk = smf.ols(formula_mturk, data=df).fit(cov_type='HC3')

# Negative Binomial on count deaths (more appropriate for counts)
# Add a small constant to ndam15 to avoid issues if any zeros (none expected, but safe)
# Use log link default
nb_formula_masfem = 'alldeaths ~ masfem + ' + ' + '.join(controls)
nb_formula_gender = 'alldeaths ~ gender_mf + ' + ' + '.join(controls)

nb_masfem = smf.glm(nb_formula_masfem, data=df, family=sm.families.NegativeBinomial()).fit()
nb_gender = smf.glm(nb_formula_gender, data=df, family=sm.families.NegativeBinomial()).fit()

# Collect key results
results = {
    'ols_masfem_coef': ols_masfem.params.get('masfem', np.nan),
    'ols_masfem_p': ols_masfem.pvalues.get('masfem', np.nan),
    'ols_gender_coef': ols_gender.params.get('gender_mf', np.nan),
    'ols_gender_p': ols_gender.pvalues.get('gender_mf', np.nan),
    'ols_mturk_coef': ols_mturk.params.get('masfem_mturk', np.nan),
    'ols_mturk_p': ols_mturk.pvalues.get('masfem_mturk', np.nan),
    'nb_masfem_coef': nb_masfem.params.get('masfem', np.nan),
    'nb_masfem_p': nb_masfem.pvalues.get('masfem', np.nan),
    'nb_gender_coef': nb_gender.params.get('gender_mf', np.nan),
    'nb_gender_p': nb_gender.pvalues.get('gender_mf', np.nan),
}

print('Key results (controls: wind, min pressure, category, ndam15, year):')
for k, v in results.items():
    print(f'{k}: {v}')

# Save results to a CSV for quick inspection
pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)
