import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
_df = _df.copy()
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Efficiency: nuts per second
_df['rate'] = _df['nuts_opened'] / _df['seconds']

print('Rows:', len(_df))
print(_df[['nuts_opened','seconds','rate']].describe())

# Poisson regression with exposure (seconds) to model counts -> efficiency
# This estimates effects on the rate of nuts opened per second
model_glm = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df['seconds'])
).fit(cov_type='HC0')

print('\nPoisson GLM with log(seconds) offset (robust SE):')
print(model_glm.summary())

# Incidence rate ratios
params = model_glm.params
irr = np.exp(params)
ci = model_glm.conf_int()
ci_irr = np.exp(ci)
print('\nIncidence Rate Ratios (IRR) with 95% CI:')
for term in irr.index:
    lo, hi = ci_irr.loc[term]
    print(f'{term:15s} IRR={irr[term]:.3f}  CI[{lo:.3f}, {hi:.3f}]')

# OLS on rate as a robustness check
model_ols = smf.ols('rate ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')
print('\nOLS on rate (nuts/sec) with robust SE:')
print(model_ols.summary())
