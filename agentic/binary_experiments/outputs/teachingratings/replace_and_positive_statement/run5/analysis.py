import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('teachingratings.csv')

# Basic correlation
corr = DF['beauty'].corr(DF['eval'])

# OLS with controls similar to literature
# Use categorical controls for instructor/course characteristics
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)
model = smf.ols(formula, data=DF).fit(cov_type='HC3')

# Also a simpler model without controls
simple = smf.ols('eval ~ beauty', data=DF).fit(cov_type='HC3')

print('Rows:', len(DF))
print('Correlation beauty vs eval:', corr)
print('\nSimple model (eval ~ beauty):')
print(simple.summary().tables[1])
print('\nControlled model:')
print(model.summary().tables[1])

# Extract key numbers for reporting
result = {
    'corr': corr,
    'simple_coef': simple.params['beauty'],
    'simple_p': simple.pvalues['beauty'],
    'controlled_coef': model.params['beauty'],
    'controlled_p': model.pvalues['beauty'],
}

print('\nKey results:', result)
