import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Clean/ensure types
# children is categorical yes/no

# Basic group stats
group_stats = df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])
group_any = (
    df.assign(any_affair=(df['affairs'] > 0).astype(int))
    .groupby('children')['any_affair']
    .agg(['mean', 'count'])
)

# t-test (Welch) for mean differences
from scipy import stats

yes = df.loc[df['children'] == 'yes', 'affairs']
no = df.loc[df['children'] == 'no', 'affairs']
t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Logistic regression for any affair with children only
df['any_affair'] = (df['affairs'] > 0).astype(int)
logit = smf.logit('any_affair ~ C(children)', data=df).fit(disp=False)

# Linear regression for affairs count with children only
ols = smf.ols('affairs ~ C(children)', data=df).fit()

# Add controls regression (commonly used covariates)
controls = (
    'C(children) + age + yearsmarried + C(gender) + religiousness + '
    'education + occupation + rating'
)
ols_ctrl = smf.ols(f'affairs ~ {controls}', data=df).fit()
logit_ctrl = smf.logit(f'any_affair ~ {controls}', data=df).fit(disp=False)

# Collect results
print('Group stats (affairs):')
print(group_stats)
print('\nAny affair rate:')
print(group_any)
print('\nWelch t-test (yes vs no): t=', t_stat, 'p=', p_val)
print('\nOLS children effect:', ols.params)
print('OLS p-values:', ols.pvalues)
print('\nLogit children effect:', logit.params)
print('Logit p-values:', logit.pvalues)
print(
    '\nOLS controls children effect:',
    ols_ctrl.params['C(children)[T.yes]'],
    'p=',
    ols_ctrl.pvalues['C(children)[T.yes]'],
)
print(
    'Logit controls children effect:',
    logit_ctrl.params['C(children)[T.yes]'],
    'p=',
    logit_ctrl.pvalues['C(children)[T.yes]'],
)

# Save summary to csv for inspection
group_stats.to_csv('group_stats.csv')
group_any.to_csv('group_any.csv')
