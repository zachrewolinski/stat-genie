import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic correlation
r, p = stats.pearsonr(df['beauty'], df['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Controlled OLS (exclude division as unique id, and students as id-like; include other covariates)
# Treat categorical columns as categorical via C().
formula_ctrl = 'allstudents ~ beauty + age + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + rownames + minority'
model_ctrl = smf.ols(formula_ctrl, data=df).fit()

# Standardized effect (beta) for beauty in simple model
beauty_std = df['beauty'].std(ddof=0)
allstudents_std = df['allstudents'].std(ddof=0)
std_beta_simple = model_simple.params['beauty'] * (beauty_std / allstudents_std)

# Effect per 1 SD beauty in original units
effect_per_sd = model_simple.params['beauty'] * beauty_std

print('n', len(df))
print('pearson_r', r)
print('pearson_p', p)
print('simple_coef', model_simple.params['beauty'])
print('simple_p', model_simple.pvalues['beauty'])
print('simple_r2', model_simple.rsquared)
print('std_beta_simple', std_beta_simple)
print('effect_per_sd', effect_per_sd)
print('ctrl_coef', model_ctrl.params['beauty'])
print('ctrl_p', model_ctrl.pvalues['beauty'])
print('ctrl_r2', model_ctrl.rsquared)

# Also compute Spearman for robustness
rho, p_s = stats.spearmanr(df['beauty'], df['allstudents'])
print('spearman_r', rho)
print('spearman_p', p_s)
