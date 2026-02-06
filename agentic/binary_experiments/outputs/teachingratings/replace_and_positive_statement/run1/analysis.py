import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic OLS: eval on beauty
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')

# OLS with controls (categorical controls encoded via C())
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Collect key results
results = {
    'simple': {
        'coef_beauty': model_simple.params.get('beauty'),
        'pvalue_beauty': model_simple.pvalues.get('beauty'),
        'nobs': int(model_simple.nobs),
        'r2': model_simple.rsquared,
    },
    'controls': {
        'coef_beauty': model_controls.params.get('beauty'),
        'pvalue_beauty': model_controls.pvalues.get('beauty'),
        'nobs': int(model_controls.nobs),
        'r2': model_controls.rsquared,
    },
}

print('Simple OLS (eval ~ beauty)')
print(results['simple'])
print('\nOLS with controls')
print(results['controls'])
