import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

df = pd.read_csv('teachingratings.csv')
print(df.head())
print(df.columns)
print(df.isna().sum())
print(df[['beauty','eval']].describe())

corr = df['beauty'].corr(df['eval'])
print('corr', corr)

m1 = smf.ols('eval ~ beauty', data=df).fit()
print(m1.summary())

formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'

m2 = smf.ols(formula, data=df).fit()
print(m2.summary())

m2_rob = smf.ols(formula, data=df).fit(cov_type="HC3")
print(m2_rob.summary())

beauty_sd = df['beauty'].std()
coef = m2_rob.params['beauty']
print('beauty sd', beauty_sd, 'coef', coef, '1sd effect', coef*beauty_sd)
