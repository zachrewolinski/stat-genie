import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Define efficiency as nuts opened per second (rate)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Primary model: Poisson GLM on counts with exposure (seconds)
# This directly models the rate while respecting count nature of nuts_opened.
_df['log_seconds'] = np.log(_df['seconds'])
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit()

# Secondary model: OLS on efficiency for a simple, direct check
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

print('Poisson GLM (rate with log(seconds) offset)')
print(poisson_model.summary())
print('\nOLS on efficiency (HC3 robust SE)')
print(ols_model.summary())

# Extract key results
results = pd.DataFrame({
    'coef': poisson_model.params,
    'pval': poisson_model.pvalues
})
print('\nPoisson GLM coefficients and p-values')
print(results)
