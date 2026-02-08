import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic derived variables
_df['has_children'] = _df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group summaries
summary = _df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any=('any_affair', 'mean')
).reset_index()

# Two-sample difference in means (affairs count)
mean_yes = _df.loc[_df['children'] == 'yes', 'affairs'].mean()
mean_no = _df.loc[_df['children'] == 'no', 'affairs'].mean()
mean_diff = mean_yes - mean_no

prop_yes = _df.loc[_df['children'] == 'yes', 'any_affair'].mean()
prop_no = _df.loc[_df['children'] == 'no', 'any_affair'].mean()
prop_diff = prop_yes - prop_no

# OLS on affairs (count-ish) with controls
ols = smf.ols(
    'affairs ~ has_children + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=_df
).fit()

# Logistic regression on any affair
logit = smf.logit(
    'any_affair ~ has_children + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=_df
).fit(disp=False)

# Poisson regression on affairs counts
poisson = smf.poisson(
    'affairs ~ has_children + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=_df
).fit(disp=False)

print('Summary by children:\n', summary.to_string(index=False))
print('\nMean affairs difference (yes - no):', mean_diff)
print('Proportion any affair difference (yes - no):', prop_diff)
print('\nOLS has_children coef:', ols.params['has_children'], 'p:', ols.pvalues['has_children'])
print('Logit has_children coef:', logit.params['has_children'], 'p:', logit.pvalues['has_children'])
print('Poisson has_children coef:', poisson.params['has_children'], 'p:', poisson.pvalues['has_children'])

# Convert logit/poisson coef to multiplicative effect
print('Logit OR:', np.exp(logit.params['has_children']))
print('Poisson IRR:', np.exp(poisson.params['has_children']))
