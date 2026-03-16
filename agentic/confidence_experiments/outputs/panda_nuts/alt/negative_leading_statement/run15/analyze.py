import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Clean: ensure categorical
for col in ['sex','help','hammer','chimpanzee']:
    df[col] = df[col].astype('category')

# Efficiency: nuts opened per second
# Avoid division by zero (seconds min 2.5, but safeguard)
df['efficiency'] = df['nuts_opened'] / df['seconds'].replace(0, np.nan)

# Basic summary
summary = df[['nuts_opened','seconds','efficiency','age']].describe()

# OLS on efficiency with predictors age, sex, help; add hammer as control
ols_formula = 'efficiency ~ age + sex + help + C(hammer)'
ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')

# Poisson regression for nuts_opened with log(seconds) offset (rate model)
# Use robust SEs for mild misspecification
poisson_formula = 'nuts_opened ~ age + sex + help + C(hammer)'
poisson_model = smf.glm(
    formula=poisson_formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit(cov_type='HC3')

# Also test without hammer to see if results stable
ols_no_hammer = smf.ols('efficiency ~ age + sex + help', data=df).fit(cov_type='HC3')
poisson_no_hammer = smf.glm(
    formula='nuts_opened ~ age + sex + help',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit(cov_type='HC3')

# Collect p-values and effects
results = {
    'ols': ols_model.summary2().tables[1],
    'poisson': poisson_model.summary2().tables[1],
    'ols_no_hammer': ols_no_hammer.summary2().tables[1],
    'poisson_no_hammer': poisson_no_hammer.summary2().tables[1],
}

# Save key stats to csv for inspection
for name, table in results.items():
    table.to_csv(f'{name}_coef.csv')

summary.to_csv('summary.csv')

print('OLS efficiency (with hammer)')
print(ols_model.summary())
print('\nPoisson rate (with hammer)')
print(poisson_model.summary())
print('\nOLS efficiency (no hammer)')
print(ols_no_hammer.summary())
print('\nPoisson rate (no hammer)')
print(poisson_no_hammer.summary())
