import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df))

# Simple correlations
masfem = df['masfem']
deaths = df['alldeaths']

# Pearson and Spearman
pearson_r, pearson_p = stats.pearsonr(masfem, deaths)
spear_r, spear_p = stats.spearmanr(masfem, deaths)
print('pearson_r', pearson_r, 'p', pearson_p)
print('spearman_r', spear_r, 'p', spear_p)

# Mean deaths by binary gender
means = df.groupby('gender_mf')['alldeaths'].mean()
print('mean deaths by gender_mf', means.to_dict())

# Log1p deaths for regression
log_deaths = np.log1p(deaths)
df = df.copy()
df['log_deaths'] = log_deaths

# OLS regression with controls
# Use category, wind, min pressure, and year as controls
model1 = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=df).fit(cov_type='HC3')
print('\nOLS log1p deaths ~ masfem + category + wind + min + year (HC3)')
print(model1.summary().tables[1])

# Alternate: binary gender
model2 = smf.ols('log_deaths ~ gender_mf + category + wind + min + year', data=df).fit(cov_type='HC3')
print('\nOLS log1p deaths ~ gender_mf + category + wind + min + year (HC3)')
print(model2.summary().tables[1])

# Simpler model: without controls
model3 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
print('\nOLS log1p deaths ~ masfem (HC3)')
print(model3.summary().tables[1])

# Robust check: Poisson with log link (counts) with controls
try:
    model4 = smf.glm(
        'alldeaths ~ masfem + category + wind + min + year',
        data=df,
        family=sm.families.Poisson()
    ).fit(cov_type='HC3')
    print('\nPoisson alldeaths ~ masfem + category + wind + min + year (HC3)')
    print(model4.summary().tables[1])
except Exception as e:
    print('Poisson model failed:', e)
