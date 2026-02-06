import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Simple bivariate model
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multivariate model with controls
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

results = {
    'simple_coef': model_simple.params['beauty'],
    'simple_pvalue': model_simple.pvalues['beauty'],
    'controls_coef': model_controls.params['beauty'],
    'controls_pvalue': model_controls.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_r2': model_controls.rsquared,
    'n': int(model_controls.nobs),
}

print(results)
