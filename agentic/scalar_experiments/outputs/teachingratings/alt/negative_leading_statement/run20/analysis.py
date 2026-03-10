import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic stats
print('rows', len(df))
print(df[['beauty','eval']].describe())

# Pearson correlation
r, p = stats.pearsonr(df['beauty'], df['eval'])
print('pearson r', r, 'p', p)

# Simple OLS
m1 = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')
print('m1 coef', m1.params['beauty'], 'p', m1.pvalues['beauty'])
print('m1 r2', m1.rsquared)

# Controls
# C() for categorical
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'
m2 = smf.ols(formula, data=df).fit(cov_type='HC3')
print('m2 coef', m2.params['beauty'], 'p', m2.pvalues['beauty'])
print('m2 r2', m2.rsquared)

# Standardized effect size: beauty 1 SD
beauty_sd = df['beauty'].std()
coef = m2.params['beauty']
print('beauty sd', beauty_sd, 'effect per 1 sd', coef*beauty_sd)

# Maybe include class size ratio? compute evaluation sample proportion
if 'students' in df.columns and 'allstudents' in df.columns:
    df['response_rate'] = df['students'] / df['allstudents']
    m3 = smf.ols(formula + ' + response_rate', data=df).fit(cov_type='HC3')
    print('m3 coef', m3.params['beauty'], 'p', m3.pvalues['beauty'])
    print('m3 r2', m3.rsquared)

