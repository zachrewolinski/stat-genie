import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric types
numeric_cols = ['masfem','masfem_mturk','gender_mf','alldeaths','wind','min','category','ndam','ndam15','year','elapsedyrs']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Outcome and predictors
# log1p deaths to handle skew and zeros
df['log_deaths'] = np.log1p(df['alldeaths'])

# Correlations
pearson = stats.pearsonr(df['masfem'], df['alldeaths'])
spearman = stats.spearmanr(df['masfem'], df['alldeaths'])

# Regression models
# Model 1: log deaths on masfem only
m1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Model 2: add storm intensity controls
# Use wind, min pressure, category; plus year to adjust long-term trends
m2 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')

# Model 3: include damage proxy (ndam15) which may capture exposure but could be post-treatment
m3 = smf.ols('log_deaths ~ masfem + wind + min + category + year + ndam15', data=df).fit(cov_type='HC3')

# Gender binary alternative
m4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit(cov_type='HC3')

# Poisson GLM with robust SE as sensitivity
glm_pois = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Negative binomial (discrete) to address overdispersion
try:
    nb = smf.negativebinomial('alldeaths ~ masfem + wind + min + category + year', data=df).fit(disp=0)
    nb_masfem = {'coef': nb.params['masfem'], 'p': nb.pvalues['masfem']}
except Exception as e:
    nb = None
    nb_masfem = {'coef': np.nan, 'p': np.nan, 'error': str(e)}

# Collect key results
results = {
    'n': int(df.shape[0]),
    'pearson_masfem_deaths': {'r': pearson.statistic, 'p': pearson.pvalue},
    'spearman_masfem_deaths': {'r': spearman.correlation, 'p': spearman.pvalue},
    'm1_masfem': {'coef': m1.params['masfem'], 'p': m1.pvalues['masfem']},
    'm2_masfem': {'coef': m2.params['masfem'], 'p': m2.pvalues['masfem']},
    'm3_masfem': {'coef': m3.params['masfem'], 'p': m3.pvalues['masfem']},
    'm4_gender_mf': {'coef': m4.params['gender_mf'], 'p': m4.pvalues['gender_mf']},
    'glm_pois_masfem': {'coef': glm_pois.params['masfem'], 'p': glm_pois.pvalues['masfem']},
    'nb_masfem': nb_masfem,
}

print(json.dumps(results, indent=2))

# Also print confidence intervals for masfem in m2
print('\nM2 masfem 95% CI:', m2.conf_int().loc['masfem'].tolist())
print('M3 masfem 95% CI:', m3.conf_int().loc['masfem'].tolist())
print('GLM Poisson masfem 95% CI:', glm_pois.conf_int().loc['masfem'].tolist())
if nb is not None:
    print('NB masfem 95% CI:', nb.conf_int().loc['masfem'].tolist())
