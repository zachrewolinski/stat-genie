import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Compute student-teacher ratio (students per teacher)
_df['stratio'] = _df['students'] / _df['teachers']

# Academic performance: average of reading and math scores
_df['avg_score'] = (_df['read'] + _df['math']) / 2

# Simple correlation
corr = _df['stratio'].corr(_df['avg_score'])

# Simple regression
X_simple = sm.add_constant(_df['stratio'])
model_simple = sm.OLS(_df['avg_score'], X_simple).fit()

# Multiple regression controlling for demographics and spending
control_vars = ['lunch', 'calworks', 'english', 'income', 'expenditure']
X_multi = sm.add_constant(_df[['stratio'] + control_vars])
model_multi = sm.OLS(_df['avg_score'], X_multi).fit()

# Save key results for inspection
results = {
    'corr_stratio_avgscore': corr,
    'simple_coef': model_simple.params['stratio'],
    'simple_pvalue': model_simple.pvalues['stratio'],
    'multi_coef': model_multi.params['stratio'],
    'multi_pvalue': model_multi.pvalues['stratio'],
}

print(results)
