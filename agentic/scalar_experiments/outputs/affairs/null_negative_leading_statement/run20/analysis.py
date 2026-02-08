import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Ensure consistent types
_df['children'] = _df['children'].astype(str).str.lower()
_df['has_children'] = (_df['children'] == 'yes').astype(int)
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Basic counts
counts = _df['has_children'].value_counts().sort_index()

# Means
mean_affairs = _df.groupby('has_children')['affairs'].mean()
mean_any = _df.groupby('has_children')['any_affair'].mean()

# Difference in means (affairs)
no = _df[_df['has_children'] == 0]['affairs']
yes = _df[_df['has_children'] == 1]['affairs']

tstat, pval = stats.ttest_ind(yes, no, equal_var=False)

# Difference in proportions (any affair)
no_any = _df[_df['has_children'] == 0]['any_affair']
yes_any = _df[_df['has_children'] == 1]['any_affair']
prop_diff = yes_any.mean() - no_any.mean()
# two-proportion z-test
count = np.array([yes_any.sum(), no_any.sum()])
obs = np.array([len(yes_any), len(no_any)])
stat_prop, pval_prop = sm.stats.proportions_ztest(count, obs)

# OLS with controls for affairs count
# Use categorical for gender and children
formula = 'affairs ~ has_children + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
model = smf.ols(formula=formula, data=_df).fit()

# Logit for any affair
logit_model = smf.logit('any_affair ~ has_children + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit(disp=False)

print('N:', len(_df))
print('Children counts (0=no,1=yes):', counts.to_dict())
print('Mean affairs (no children, children):', mean_affairs.to_dict())
print('Mean any affair (no children, children):', mean_any.to_dict())
print('Affairs mean diff (children - no children):', yes.mean() - no.mean())
print('t-test (affairs): t=%.4f p=%.6f' % (tstat, pval))
print('Any affair prop diff (children - no children):', prop_diff)
print('Prop z-test: z=%.4f p=%.6f' % (stat_prop, pval_prop))
print('\nOLS coefficient for has_children (affairs):')
print(model.params['has_children'], model.pvalues['has_children'])
print('\nLogit coefficient for has_children (any affair):')
print(logit_model.params['has_children'], logit_model.pvalues['has_children'])

# Also report marginal effect (odds ratio)
odds_ratio = np.exp(logit_model.params['has_children'])
print('Odds ratio (has_children):', odds_ratio)
