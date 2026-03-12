import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Rename columns for clarity based on value ranges
# year_col= wind (1950-2012)
# name_col = alldeaths (string)
# fem_index = category (1-11)
# min_pressure = ndam15 (909-1002)
# female_binary = masfem_mturk (0/1)
# storm_cat = gender_mf (1-5)
# deaths = name (0-1833)
# damage1 = elapsedyrs (1-75000)
# years_elapsed = masfem (1-63)
# mturk_index = ind (1-11)
# max_wind = year (75-190)
# damage2 = source (1-88420)

# add derived columns
_df = _df.copy()
_df['log_deaths'] = np.log1p(_df['name'])
_df['log_damage1'] = np.log1p(_df['elapsedyrs'])
_df['log_damage2'] = np.log1p(_df['source'])

# Basic correlations
corr_fem_deaths = _df[['category','ind','masfem_mturk','name','log_deaths']].corr(numeric_only=True)
print('Correlation (fem indices vs deaths):')
print(corr_fem_deaths)

# OLS regression: log deaths ~ fem_index + controls
# Controls: max wind, min pressure, Saffir-Simpson category, damage2 (log)
model1 = smf.ols('log_deaths ~ category + year + ndam15 + gender_mf + log_damage2', data=_df).fit(cov_type='HC3')
print('\nOLS log_deaths ~ fem_index + controls')
print(model1.summary())

# Alternative with mturk index and binary
model2 = smf.ols('log_deaths ~ ind + year + ndam15 + gender_mf + log_damage2', data=_df).fit(cov_type='HC3')
print('\nOLS log_deaths ~ mturk index + controls')
print(model2.summary())

model3 = smf.ols('log_deaths ~ masfem_mturk + year + ndam15 + gender_mf + log_damage2', data=_df).fit(cov_type='HC3')
print('\nOLS log_deaths ~ female_binary + controls')
print(model3.summary())

# Poisson regression for deaths (count) with controls
poisson1 = smf.glm('name ~ category + year + ndam15 + gender_mf + log_damage2', data=_df, family=sm.families.Poisson()).fit()
print('\nPoisson deaths ~ fem_index + controls')
print(poisson1.summary())

# Check interaction (fem_index * severity) with wind speed
model_int = smf.ols('log_deaths ~ category * year + ndam15 + gender_mf + log_damage2', data=_df).fit(cov_type='HC3')
print('\nOLS log_deaths ~ fem_index * wind + controls')
print(model_int.summary())

# Save key coefficients for review
for name, m in [('model1', model1), ('model2', model2), ('model3', model3), ('poisson1', poisson1), ('model_int', model_int)]:
    print('\n', name)
    print(m.params)
    print(m.pvalues)

