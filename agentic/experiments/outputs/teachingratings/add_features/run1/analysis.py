import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Use core columns for teaching evaluations
cols = [
    'eval', 'beauty', 'age', 'gender', 'minority', 'credits', 'division',
    'native', 'tenure', 'students', 'allstudents'
]

df = _df[cols].dropna().copy()

# Treat categorical variables as categories
cat_cols = ['gender', 'minority', 'credits', 'division', 'native', 'tenure']
for c in cat_cols:
    df[c] = df[c].astype('category')

# OLS regression with controls
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)

model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

# Save key results for reporting
result = {
    'n': int(model.nobs),
    'beauty_coef': float(model.params['beauty']),
    'beauty_pvalue': float(model.pvalues['beauty']),
    'beauty_ci_low': float(model.conf_int().loc['beauty', 0]),
    'beauty_ci_high': float(model.conf_int().loc['beauty', 1]),
    'r2': float(model.rsquared),
}

# Print a concise summary for inspection
print('N:', result['n'])
print('Beauty coef:', result['beauty_coef'])
print('Beauty p-value:', result['beauty_pvalue'])
print('Beauty 95% CI:', (result['beauty_ci_low'], result['beauty_ci_high']))
print('R^2:', result['r2'])
