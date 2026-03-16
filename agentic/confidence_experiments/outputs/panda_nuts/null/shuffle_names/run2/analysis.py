import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Map columns based on info.json descriptions (names were shuffled)
# actual age -> column 'hammer'
# actual sex -> column 'nuts_opened' (f/m)
# actual help (received help) -> column 'seconds' (y/N)
# nuts opened count -> column 'help'
# session duration in seconds -> column 'chimpanzee'

# Compute efficiency: nuts opened per second

df = df.copy()
df['efficiency'] = df['help'] / df['chimpanzee']

# Recode variables

df['age_years'] = df['hammer']
df['sex'] = df['nuts_opened'].astype('category')
df['received_help'] = df['seconds'].astype('category')

# Basic checks
print('Rows:', len(df))
print(df[['age_years','sex','received_help','help','chimpanzee','efficiency']].describe(include='all'))

# OLS regression with categorical predictors
model = smf.ols('efficiency ~ age_years + C(sex) + C(received_help)', data=df).fit(cov_type='HC3')
print(model.summary())

# Joint F-test for all predictors
joint_test = model.f_test('age_years = 0, C(sex)[T.m] = 0, C(received_help)[T.y] = 0')
print('\nJoint F-test (all predictors):')
print(joint_test)

# Also compute group means for interpretation
print('\nGroup means by sex:')
print(df.groupby('sex')['efficiency'].agg(['mean','count','std']))
print('\nGroup means by received_help:')
print(df.groupby('received_help')['efficiency'].agg(['mean','count','std']))

# Correlation with age
print('\nCorrelation age vs efficiency:')
print(df[['age_years','efficiency']].corr().iloc[0,1])
