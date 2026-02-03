import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic sanity checks
print('Rows:', len(_df))
print('Columns:', list(_df.columns))

# Simple correlation
corr = _df['beauty'].corr(_df['eval'])
print(f'Correlation(beauty, eval): {corr:.4f}')

# Simple regression
m1 = smf.ols('eval ~ beauty', data=_df).fit()
print('\nModel 1: eval ~ beauty')
print(m1.summary())

# Multiple regression with key controls
# Categorical variables as factors
formula = (
    'eval ~ beauty + age + students + allstudents + '
    'C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
)

m2 = smf.ols(formula, data=_df).fit()
print('\nModel 2: eval ~ beauty + controls')
print(m2.summary())

# Clustered by professor (optional robustness) using prof id
try:
    m2_cluster = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prof']})
    print('\nModel 2 (clustered SE by prof):')
    print(m2_cluster.summary())
except Exception as e:
    print('Clustered SE failed:', e)

# Extract key effect sizes
b1 = m1.params['beauty']
p1 = m1.pvalues['beauty']

b2 = m2.params['beauty']
p2 = m2.pvalues['beauty']

print('\nKey results:')
print(f'Unadjusted: beauty coef={b1:.4f}, p={p1:.4g}')
print(f'Adjusted:   beauty coef={b2:.4f}, p={p2:.4g}')
