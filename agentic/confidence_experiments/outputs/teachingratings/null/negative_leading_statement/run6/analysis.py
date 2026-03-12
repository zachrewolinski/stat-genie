import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic info
print('Rows:', len(df))
print('Columns:', df.columns.tolist())

# Correlation between beauty and eval
corr = df['beauty'].corr(df['eval'])
print('Pearson corr beauty vs eval:', corr)

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df).fit()
print('\nSimple OLS: eval ~ beauty')
print(model_simple.summary())

# Multiple OLS with plausible controls (based on dataset variables)
# Convert categorical to C() for formula
controls = ' + '.join([
    'C(gender)', 'C(minority)', 'age', 'C(credits)', 'C(division)', 'C(native)',
    'C(tenure)', 'students', 'allstudents'
])
formula = f'eval ~ beauty + {controls}'
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')
print('\nOLS with controls (HC3 robust SEs):', formula)
print(model_controls.summary())

# Also consider instructor fixed effects? can't with prof as category due to 94 instructors.
# But we can include C(prof) to control instructor-level differences.
# However, eval is per course and beauty is per instructor, so FE may absorb beauty.
# We'll report that as a robustness check (beauty likely collinear with C(prof)).
try:
    model_fe = smf.ols('eval ~ beauty + C(prof)', data=df).fit(cov_type='HC3')
    print('\nOLS with instructor fixed effects (HC3): eval ~ beauty + C(prof)')
    print(model_fe.summary())
except Exception as e:
    print('FE model failed:', e)

# Compute standardized effect size for beauty in simple model
# standardize both
z_beauty = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()
zeval = (df['eval'] - df['eval'].mean()) / df['eval'].std()
model_std = smf.ols('zeval ~ z_beauty', data=pd.DataFrame({'zeval': zeval, 'z_beauty': z_beauty})).fit()
print('\nStandardized OLS: zeval ~ z_beauty')
print(model_std.summary())

# Save key stats for later inspection
print('\nKey stats:')
print('Simple model beauty coef:', model_simple.params['beauty'], 'p:', model_simple.pvalues['beauty'])
print('Controls model beauty coef:', model_controls.params['beauty'], 'p:', model_controls.pvalues['beauty'])
print('Std model beta:', model_std.params['z_beauty'], 'p:', model_std.pvalues['z_beauty'])
