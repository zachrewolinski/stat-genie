import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('hurricane.csv')

# Outcome transformations

df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic correlations
corr = df[['masfem','gender_mf','alldeaths','log_deaths','wind','min','category']].corr(numeric_only=True)
print('correlations')
print(corr)

# OLS with robust SEs
ols1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
ols2 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')
ols3 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit(cov_type='HC3')

print('\nOLS1')
print(ols1.summary().tables[1])
print('\nOLS2')
print(ols2.summary().tables[1])
print('\nOLS3')
print(ols3.summary().tables[1])

# Poisson GLM (log link) with robust SEs
poisson = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df,
                  family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson')
print(poisson.summary().tables[1])

# Negative binomial GLM
try:
    nb = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df,
                 family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    print('\nNegBin')
    print(nb.summary().tables[1])
except Exception as e:
    print('NegBin failed', e)

# Simple group comparison for gender_mf (female vs male)
# use log_deaths
female = df.loc[df['gender_mf'] == 1, 'log_deaths']
male = df.loc[df['gender_mf'] == 0, 'log_deaths']

t_stat, p_val = stats.ttest_ind(female, male, equal_var=False)
print('\nT-test log_deaths female vs male')
print({'t': t_stat, 'p': p_val, 'female_mean': female.mean(), 'male_mean': male.mean()})

# Save key stats for conclusion
results = {
    'ols1_masfem_coef': float(ols1.params['masfem']),
    'ols1_masfem_p': float(ols1.pvalues['masfem']),
    'ols2_masfem_coef': float(ols2.params['masfem']),
    'ols2_masfem_p': float(ols2.pvalues['masfem']),
    'ols3_gender_coef': float(ols3.params['gender_mf']),
    'ols3_gender_p': float(ols3.pvalues['gender_mf']),
    'poisson_masfem_coef': float(poisson.params['masfem']),
    'poisson_masfem_p': float(poisson.pvalues['masfem']),
}
try:
    results['nb_masfem_coef'] = float(nb.params['masfem'])
    results['nb_masfem_p'] = float(nb.pvalues['masfem'])
except Exception:
    pass

print('\nKEY_RESULTS')
print(results)
