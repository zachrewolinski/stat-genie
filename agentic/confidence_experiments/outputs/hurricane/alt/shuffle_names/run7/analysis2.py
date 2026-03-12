import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

pd.set_option('display.max_columns', 80)

# Load data

df = pd.read_csv('hurricane.csv')

# Variables

df['fatalities'] = df['name']

df['log_fatalities'] = np.log1p(df['fatalities'])
df['log_damage_2013'] = np.log1p(df['elapsedyrs'])
df['log_damage_2015'] = np.log1p(df['source'])

# Define predictors
predictors_basic = ['category', 'year', 'ndam15', 'gender_mf']
predictors_plus_damage = predictors_basic + ['log_damage_2013', 'wind']

# OLS reference
ols_formula = 'log_fatalities ~ ' + ' + '.join(predictors_plus_damage)
ols = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
print('OLS (log fatalities) with controls:')
print(ols.summary().tables[1])

# Poisson GLM with robust SE
poisson_formula = 'fatalities ~ ' + ' + '.join(predictors_plus_damage)
poisson = smf.glm(poisson_formula, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson GLM with controls (robust SE):')
print(poisson.summary().tables[1])

# Negative binomial GLM with fixed alpha (1.0)
nb_glm = smf.glm(poisson_formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit(cov_type='HC3')
print('\nNegative Binomial GLM (alpha=1) with controls:')
print(nb_glm.summary().tables[1])

# Negative binomial (discrete) with estimated alpha
# Add constant manually
X = df[predictors_plus_damage].copy()
X = sm.add_constant(X, has_constant='add')

y = df['fatalities']
nb2 = NegativeBinomial(y, X).fit(disp=False)
print('\nNegative Binomial (discrete) with estimated alpha:')
print(nb2.summary().tables[1])
print('Estimated alpha:', nb2.params.get('alpha'))

# Check model with MTurk rating (ind)
X2 = df[['ind', 'year', 'ndam15', 'gender_mf', 'log_damage_2013', 'wind']].copy()
X2 = sm.add_constant(X2, has_constant='add')
nb2_ind = NegativeBinomial(y, X2).fit(disp=False)
print('\nNegative Binomial (discrete) with MTurk rating:')
print(nb2_ind.summary().tables[1])

