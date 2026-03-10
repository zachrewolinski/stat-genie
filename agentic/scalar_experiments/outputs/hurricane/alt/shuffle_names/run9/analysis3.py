import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

analysis = df.copy()
analysis['log_deaths'] = np.log1p(analysis['name'])

# Bivariate OLS
model_cat = smf.ols('log_deaths ~ category', data=analysis).fit(cov_type='HC3')
model_ind = smf.ols('log_deaths ~ ind', data=analysis).fit(cov_type='HC3')

print('Bivariate category:', model_cat.params['category'], model_cat.pvalues['category'], 'R2', model_cat.rsquared)
print('Bivariate ind:', model_ind.params['ind'], model_ind.pvalues['ind'], 'R2', model_ind.rsquared)

# Spearman correlations (robust to nonlinearity)
from scipy.stats import spearmanr

spearman_cat = spearmanr(analysis['category'], analysis['name'])
spearman_ind = spearmanr(analysis['ind'], analysis['name'])
print('Spearman category vs deaths:', spearman_cat)
print('Spearman ind vs deaths:', spearman_ind)
