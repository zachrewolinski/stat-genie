import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('hurricane.csv')
df['log_deaths'] = np.log1p(df['alldeaths'])

# OLS with robust SE
m2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC1')

# Negative binomial (GLM) with robust SE
nb = smf.glm('alldeaths ~ masfem + wind + min + category', data=df,
             family=sm.families.NegativeBinomial()).fit(cov_type='HC1')

# Poisson with robust SE
pois = smf.glm('alldeaths ~ masfem + wind + min + category', data=df,
               family=sm.families.Poisson()).fit(cov_type='HC1')

corr, pval = stats.pearsonr(df['masfem'], df['log_deaths'])

results = {
    'ols_robust': {
        'coef': float(m2.params['masfem']),
        'se': float(m2.bse['masfem']),
        'pval': float(m2.pvalues['masfem']),
        'r2': float(m2.rsquared)
    },
    'nb_robust': {
        'coef': float(nb.params['masfem']),
        'se': float(nb.bse['masfem']),
        'pval': float(nb.pvalues['masfem']),
        'aic': float(nb.aic)
    },
    'pois_robust': {
        'coef': float(pois.params['masfem']),
        'se': float(pois.bse['masfem']),
        'pval': float(pois.pvalues['masfem'])
    },
    'corr_log_deaths': {'r': float(corr), 'pval': float(pval)},
}

print(json.dumps(results, indent=2))
