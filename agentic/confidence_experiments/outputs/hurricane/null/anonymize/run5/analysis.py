import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.describe(include='all'))

# define variables
# feature4: femininity index, feature6: binary gender (0 male 1 female)
# feature8: fatalities
# feature5: min pressure, feature7: category, feature13: max wind speed, feature9/feature14: damages

# create log fatalities
for col in ['feature8','feature9','feature14']:
    df[f'log_{col}'] = np.log1p(df[col])

# simple correlation
print('corr fem index vs log fatalities', df['feature4'].corr(df['log_feature8']))
print('corr gender binary vs log fatalities', df['feature6'].corr(df['log_feature8']))

# linear regression controlling for severity
# Use log fatalities as response; predictors: femininity index, log_damage (feature9 or feature14), pressure, wind, category

# use feature14 (damage normalized 2015) maybe more complete

model = smf.ols('log_feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df).fit()
print(model.summary())

# interaction between femininity and severity (wind)
model_int = smf.ols('log_feature8 ~ feature4 * feature13 + log_feature14 + feature5 + feature7', data=df).fit()
print(model_int.summary())

# use gender binary
model_bin = smf.ols('log_feature8 ~ feature6 + log_feature14 + feature5 + feature13 + feature7', data=df).fit()
print(model_bin.summary())

# Poisson regression for fatalities count
poisson = smf.glm('feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df, family=sm.families.Poisson()).fit()
print(poisson.summary())

# negative binomial if available
try:
    nb = smf.glm('feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df, family=sm.families.NegativeBinomial()).fit()
    print(nb.summary())
except Exception as e:
    print('NB failed', e)
