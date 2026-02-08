import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic comparison: mean affairs and proportion any affairs by children
_df['has_affair'] = (_df['affairs'] > 0).astype(int)

summary = _df.groupby('children').agg(
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    any_affair_rate=('has_affair', 'mean'),
    n=('affairs', 'size')
)

# Encode children (yes=1, no=0)
_df['children_yes'] = (_df['children'] == 'yes').astype(int)

# OLS for affairs count (not ideal distribution, but direction/association)
ols = smf.ols('affairs ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating', data=_df).fit()

# Logistic regression for any affair
logit = smf.logit('has_affair ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating', data=_df).fit(disp=False)

# Extract key effects
ols_coef = ols.params['children_yes']
ols_p = ols.pvalues['children_yes']

logit_coef = logit.params['children_yes']
logit_p = logit.pvalues['children_yes']
logit_or = np.exp(logit_coef)

# Save results for quick inspection
print('SUMMARY')
print(summary)
print('\nOLS children_yes coef:', ols_coef, 'p=', ols_p)
print('LOGIT children_yes coef:', logit_coef, 'OR=', logit_or, 'p=', logit_p)
