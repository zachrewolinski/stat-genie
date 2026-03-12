import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))

# basic correlations with deaths
for col in ['feature4','feature12','feature6','feature7','feature5','feature13','feature9','feature14']:
    if col in df.columns:
        corr = df[col].corr(df['feature8'])
        print('corr', col, corr)

# log1p deaths

df['log_deaths'] = np.log1p(df['feature8'])

# OLS models
# Model 1: log deaths ~ femininity index
model1 = smf.ols('log_deaths ~ feature4', data=df).fit(cov_type='HC3')
print(model1.summary())

# Model 2: add storm severity controls
model2 = smf.ols('log_deaths ~ feature4 + feature7 + feature5 + feature13', data=df).fit(cov_type='HC3')
print(model2.summary())

# Model 3: add damage as proxy exposure
model3 = smf.ols('log_deaths ~ feature4 + feature7 + feature5 + feature13 + np.log1p(feature14)', data=df).fit(cov_type='HC3')
print(model3.summary())

# binary gender
model4 = smf.ols('log_deaths ~ feature6 + feature7 + feature5 + feature13', data=df).fit(cov_type='HC3')
print(model4.summary())

# Spearman correlations
from scipy.stats import spearmanr
for col in ['feature4','feature12','feature6']:
    rho, p = spearmanr(df[col], df['feature8'])
    print('spearman', col, rho, p)
