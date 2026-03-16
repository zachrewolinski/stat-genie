import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Map columns to conceptual variables
# tooth_class column contains genus categories
# sockets column contains tooth class (Anterior/Posterior/Premolar)
# age column appears to be number of observable sockets per class
# pop column appears to be age at death
# stdev_age column appears to be sex probability (0-1)

# Create outcome: AMTL rate per class
# If genus is missing count (noisy), compute rate

# avoid division by zero
rate = df['genus'] / df['age']

df = df.assign(amtl_rate=rate,
               is_human=(df['tooth_class'] == 'Homo sapiens').astype(int),
               age_at_death=df['pop'],
               sex_prob=df['stdev_age'],
               tooth_cls=df['sockets'])

# OLS regression
model = smf.ols('amtl_rate ~ is_human + age_at_death + sex_prob + C(tooth_cls)', data=df).fit(cov_type='HC3')

print(model.summary())

# also try using amtl_count as response with sockets as covariate
model2 = smf.ols('genus ~ is_human + age_at_death + sex_prob + C(tooth_cls) + age', data=df).fit(cov_type='HC3')
print('\nModel2 (count proxy)')
print(model2.summary())

