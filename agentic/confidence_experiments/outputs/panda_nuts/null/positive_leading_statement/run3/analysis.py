import pandas as pd
import statsmodels.formula.api as smf
import numpy as np


df = pd.read_csv('panda_nuts.csv')
# compute efficiency: nuts opened per second
# guard against zero seconds (none expected)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# basic info
print('rows', len(df))
print('efficiency summary')
print(df['efficiency'].describe())

# OLS with controls for hammer type
model = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit(cov_type='HC3')
print('\nOLS efficiency ~ age + sex + help + hammer (HC3 robust)')
print(model.summary())

# OLS without hammer
model2 = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nOLS efficiency ~ age + sex + help (HC3 robust)')
print(model2.summary())

# Alternative: count model with offset for seconds (Poisson) for robustness
# Add small constant to avoid zero seconds, not needed
try:
    import statsmodels.api as sm
    df['log_seconds'] = np.log(df['seconds'])
    poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help) + C(hammer)', data=df,
                      family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='HC3')
    print('\nPoisson (rate) nuts_opened with offset log(seconds)')
    print(poisson.summary())
except Exception as e:
    print('Poisson model failed', e)

