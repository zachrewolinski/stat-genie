import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Efficiency as rate of nuts opened per second
# Use a Poisson GLM with log(seconds) as an offset
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
)
poisson_res = poisson_model.fit()

# Also fit OLS on rate for a simple robustness check
# (not used for inference in the conclusion, but helpful context)
df['rate'] = df['nuts_opened'] / df['seconds']
ols_res = smf.ols('rate ~ age + C(sex) + C(help)', data=df).fit()

# Print key results for inspection
print("Poisson GLM with offset (nuts_opened ~ predictors, offset log(seconds))")
print(poisson_res.summary())
print("\nOLS on rate (nuts_opened/seconds)")
print(ols_res.summary())
