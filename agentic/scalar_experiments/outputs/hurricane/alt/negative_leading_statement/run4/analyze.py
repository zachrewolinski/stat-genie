import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Inspect
print(df.head())
print(df.describe(include='all'))

# Basic correlations
for col in ['masfem','masfem_mturk','gender_mf']:
    if col in df.columns:
        corr = df[col].corr(df['alldeaths'])
        print('corr alldeaths vs', col, corr)

# log deaths

df['log_deaths'] = np.log1p(df['alldeaths'])

# baseline linear regression: log_deaths ~ masfem + wind + min + category + ndam15?
# Use wind and min (pressure) and category

model1 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit()
print(model1.summary())

model2 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category', data=df).fit()
print(model2.summary())

model3 = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=df).fit()
print(model3.summary())

# model without controls
model4 = smf.ols('log_deaths ~ masfem', data=df).fit()
print(model4.summary())

# Alternative: Poisson or negative binomial on alldeaths

poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=df, family=sm.families.Poisson()).fit()
print(poisson.summary())

# also check if masfem interacts with wind or category (optional)

model_int = smf.ols('log_deaths ~ masfem * wind + min + category', data=df).fit()
print(model_int.summary())
