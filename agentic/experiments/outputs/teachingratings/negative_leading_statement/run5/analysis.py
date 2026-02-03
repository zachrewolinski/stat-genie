import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic checks
print('Rows, cols:', _df.shape)
print(_df[['beauty', 'eval']].describe())
print('Correlation (beauty, eval):')
print(_df[['beauty', 'eval']].corr())

# Simple regression: eval on beauty
m1 = smf.ols('eval ~ beauty', data=_df).fit()
print('\nSimple OLS: eval ~ beauty')
print(m1.summary())

# Regression with controls
_df['log_students'] = np.log(_df['students'])

m2 = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + log_students',
    data=_df
).fit()
print('\nOLS with controls')
print(m2.summary())

# Extract key stats for reporting
print('\nKey estimates:')
print('beauty (simple): coef=%.4f, p=%.4g' % (m1.params['beauty'], m1.pvalues['beauty']))
print('beauty (controls): coef=%.4f, p=%.4g' % (m2.params['beauty'], m2.pvalues['beauty']))
