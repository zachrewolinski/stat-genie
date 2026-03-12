import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.columns)
print('n', len(df))

# Create log deaths
df['log_deaths'] = np.log1p(df['alldeaths'])

# Correlations
print('corr masfem vs deaths', df['masfem'].corr(df['alldeaths']))
print('corr masfem vs log_deaths', df['masfem'].corr(df['log_deaths']))

# T-test male vs female
male = df.loc[df['gender_mf'] == 0, 'log_deaths']
female = df.loc[df['gender_mf'] == 1, 'log_deaths']
print('male mean', male.mean(), 'female mean', female.mean())
print(stats.ttest_ind(male, female, equal_var=False))

# Regression with controls
model1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
print(model1.summary())

model2 = smf.ols('log_deaths ~ masfem + category + wind + min', data=df).fit(cov_type='HC3')
print(model2.summary())

model3 = smf.ols('log_deaths ~ gender_mf + category + wind + min', data=df).fit(cov_type='HC3')
print(model3.summary())

model4 = smf.ols('log_deaths ~ masfem + category + wind + min + ndam15', data=df).fit(cov_type='HC3')
print(model4.summary())
