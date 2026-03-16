import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

pd.set_option('display.max_columns', 80)

df = pd.read_csv('hurricane.csv')

df['fatalities'] = df['name']
df['log_fatalities'] = np.log1p(df['fatalities'])
df['log_damage_2013'] = np.log1p(df['elapsedyrs'])
df['log_damage_2015'] = np.log1p(df['source'])

# Predictors
predictors_basic = ['category', 'year', 'ndam15', 'gender_mf']
predictors_plus_damage = predictors_basic + ['log_damage_2013', 'wind']

# Drop rows with missing values in required columns
needed_cols = ['fatalities', 'log_fatalities'] + predictors_plus_damage
clean = df[needed_cols].dropna().copy()

print('Rows used:', len(clean), 'out of', len(df))

# OLS
ols_formula = 'log_fatalities ~ ' + ' + '.join(predictors_plus_damage)
ols = smf.ols(ols_formula, data=clean).fit(cov_type='HC3')
print('\nOLS (log fatalities) with controls:')
print(ols.summary().tables[1])

# Poisson GLM
poisson_formula = 'fatalities ~ ' + ' + '.join(predictors_plus_damage)
poisson = smf.glm(poisson_formula, data=clean, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson GLM with controls (robust SE):')
print(poisson.summary().tables[1])

# Negative Binomial GLM with alpha=1
nb_glm = smf.glm(poisson_formula, data=clean, family=sm.families.NegativeBinomial(alpha=1.0)).fit(cov_type='HC3')
print('\nNegative Binomial GLM (alpha=1) with controls:')
print(nb_glm.summary().tables[1])

# Negative Binomial (discrete) with estimated alpha
X = clean[predictors_plus_damage].copy()
X = sm.add_constant(X, has_constant='add')

y = clean['fatalities']
nb2 = NegativeBinomial(y, X).fit(disp=False)
print('\nNegative Binomial (discrete) with estimated alpha:')
print(nb2.summary().tables[1])
print('Estimated alpha:', nb2.params.get('alpha'))

# Alternative femininity measure (ind)
# Build separate clean dataset for ind
predictors_ind = ['ind', 'year', 'ndam15', 'gender_mf', 'log_damage_2013', 'wind']
clean_ind = df[['fatalities'] + predictors_ind].dropna().copy()
X2 = sm.add_constant(clean_ind[predictors_ind], has_constant='add')
nb2_ind = NegativeBinomial(clean_ind['fatalities'], X2).fit(disp=False)
print('\nNegative Binomial (discrete) with MTurk rating:')
print(nb2_ind.summary().tables[1])

# OLS with ind
ols_formula_ind = 'np.log1p(fatalities) ~ ' + ' + '.join(predictors_ind)
ols_ind = smf.ols(ols_formula_ind, data=clean_ind).fit(cov_type='HC3')
print('\nOLS (log fatalities) with MTurk rating:')
print(ols_ind.summary().tables[1])

