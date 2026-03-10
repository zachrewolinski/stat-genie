import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# load
df = pd.read_csv('hurricane.csv')

# log deaths
df['log_deaths'] = np.log1p(df['alldeaths'])

# interaction with category and wind
formula = 'log_deaths ~ masfem * category + wind + min + ndam15 + year'
model = smf.ols(formula, data=df).fit()
print('OLS interaction masfem*category')
print(model.summary())

formula2 = 'log_deaths ~ masfem * wind + category + min + ndam15 + year'
model2 = smf.ols(formula2, data=df).fit()
print('OLS interaction masfem*wind')
print(model2.summary())

# Negative binomial interaction masfem*category
formula_nb = 'alldeaths ~ masfem * category + wind + min + ndam15 + year'
nb = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
print('NB interaction masfem*category')
print(nb.summary())

# Negative binomial interaction masfem*wind
formula_nb2 = 'alldeaths ~ masfem * wind + category + min + ndam15 + year'
nb2 = smf.glm(formula_nb2, data=df, family=sm.families.NegativeBinomial()).fit()
print('NB interaction masfem*wind')
print(nb2.summary())
