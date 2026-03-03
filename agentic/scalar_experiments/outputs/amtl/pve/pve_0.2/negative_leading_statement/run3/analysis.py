import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic checks
print('rows', len(_df))
print(_df.head())
print(_df['num_amtl'].describe())
print('num_amtl integer-like share', np.mean(np.isclose(_df['num_amtl'], np.round(_df['num_amtl']))))

# Create indicator for Homo sapiens
_df['is_homo'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Use OLS with categorical tooth_class and genus (or is_homo)
# Model 1: genus categories (Homo vs others)
model1 = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')
print(model1.summary())

# Also model with genus categories as categorical for comparison
model2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')
print(model2.summary())

# Compute adjusted mean difference: Homo vs non-human using model1
coef = model1.params['is_homo']
se = model1.bse['is_homo']
print('is_homo coef', coef, 'se', se, 'p', model1.pvalues['is_homo'])

# For interpretability, compute effect size relative to sd
sd = _df['num_amtl'].std()
print('std num_amtl', sd, 'coef/SD', coef/sd)

# Check with weights by sockets (if relevant) as a sensitivity (WLS)
# Use sockets as weights (more sockets -> more reliable)
model3 = smf.wls('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=_df, weights=_df['sockets']).fit(cov_type='HC3')
print(model3.summary())
print('WLS is_homo', model3.params['is_homo'], model3.bse['is_homo'], model3.pvalues['is_homo'])

# Another sensitivity: include sockets as covariate
model4 = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class) + sockets', data=_df).fit(cov_type='HC3')
print(model4.summary())
print('OLS+sockets is_homo', model4.params['is_homo'], model4.bse['is_homo'], model4.pvalues['is_homo'])

# Summaries by genus
print(_df.groupby('genus')['num_amtl'].agg(['mean','std','count']))

