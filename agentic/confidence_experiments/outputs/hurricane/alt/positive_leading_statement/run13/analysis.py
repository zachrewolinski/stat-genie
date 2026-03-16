import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('hurricane.csv')

# Basic cleaning
# log1p deaths
DF['log_deaths'] = np.log1p(DF['alldeaths'])

# Standardize some predictors for comparability
for col in ['masfem', 'wind', 'min', 'category', 'year', 'ndam', 'ndam15']:
    if col in DF.columns:
        DF[f'z_{col}'] = (DF[col] - DF[col].mean()) / DF[col].std(ddof=0)

# Model 1: log_deaths ~ masfem (simple)
model1 = smf.ols('log_deaths ~ masfem', data=DF).fit(cov_type='HC3')

# Model 2: control for severity (wind, min pressure, category)
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=DF).fit(cov_type='HC3')

# Model 3: control for severity + year (to capture improvements)
model3 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=DF).fit(cov_type='HC3')

# Alternative: use gender_mf (binary) with controls
model4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=DF).fit(cov_type='HC3')

# Correlations
corr = DF[['masfem', 'alldeaths', 'log_deaths', 'wind', 'min', 'category']].corr()

print('N', len(DF))
print('\nModel1 (log_deaths ~ masfem):')
print(model1.summary().tables[1])
print('\nModel2 (controls):')
print(model2.summary().tables[1])
print('\nModel3 (controls + year):')
print(model3.summary().tables[1])
print('\nModel4 (gender_mf + controls + year):')
print(model4.summary().tables[1])
print('\nCorrelations:')
print(corr)
