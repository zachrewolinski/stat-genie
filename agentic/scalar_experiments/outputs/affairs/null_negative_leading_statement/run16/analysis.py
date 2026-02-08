import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Clean/encode
_df['children_yes'] = (_df['children'].str.lower() == 'yes').astype(int)

# Outcome 1: affairs count
# Outcome 2: any affairs
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Basic group stats
summary = _df.groupby('children')['affairs'].agg(['count','mean','median'])
summary_any = _df.groupby('children')['any_affair'].agg(['mean'])

# Regression: OLS on affairs count with controls
# Use categorical for gender; other variables numeric as provided
formula_ols = 'affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)'
ols = smf.ols(formula=formula_ols, data=_df).fit()

# Logistic regression for any affair
formula_logit = 'any_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)'
logit = smf.logit(formula=formula_logit, data=_df).fit(disp=False)

# Extract effect sizes
ols_coef = float(ols.params['children_yes'])
ols_p = float(ols.pvalues['children_yes'])

logit_coef = float(logit.params['children_yes'])
logit_p = float(logit.pvalues['children_yes'])

# Convert logit coef to odds ratio for interpretability
odds_ratio = float(np.exp(logit_coef))

# Save results to a small csv for reference
out = {
    'children_group_summary': summary,
    'children_any_summary': summary_any,
}

summary.to_csv('children_affairs_summary.csv')
summary_any.to_csv('children_any_summary.csv')

# Print key numbers
print('Group summary (affairs):')
print(summary)
print('\nGroup summary (any affair rate):')
print(summary_any)
print('\nOLS coef children_yes:', ols_coef, 'p=', ols_p)
print('Logit coef children_yes:', logit_coef, 'p=', logit_p, 'odds_ratio=', odds_ratio)
