import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('affairs.csv')

# Map children to 1/0
children = df['feature6'].map({'yes': 1, 'no': 0})

df = df.assign(children=children)

# Outcome: affair frequency (feature2) and any affair indicator
outcome = df['feature2'].astype(float)
any_affair = (outcome > 0).astype(int)

# Group stats
summary = df.groupby('children')['feature2'].agg(['count', 'mean', 'median'])

# Two-sample t-test (Welch)
no_vals = outcome[df['children'] == 0]
yes_vals = outcome[df['children'] == 1]
t_stat, p_val = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy='omit')

# Proportion test for any affair
prop_no = any_affair[df['children'] == 0].mean()
prop_yes = any_affair[df['children'] == 1].mean()

# z-test for proportions
n_no = (df['children'] == 0).sum()
n_yes = (df['children'] == 1).sum()
count_no = any_affair[df['children'] == 0].sum()
count_yes = any_affair[df['children'] == 1].sum()

p_pool = (count_no + count_yes) / (n_no + n_yes)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_no + 1 / n_yes))
if se > 0:
    z = (prop_yes - prop_no) / se
    p_prop = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_prop = np.nan

# Regression controls
# Encode gender
female = (df['feature3'] == 'female').astype(int)

X = pd.DataFrame({
    'children': df['children'],
    'female': female,
    'age': df['feature4'],
    'yrs_married': df['feature5'],
    'religious': df['feature7'],
    'education': df['feature8'],
    'occupation': df['feature9'],
    'marriage_rating': df['feature10'],
})
X = sm.add_constant(X)

# OLS on affair frequency
ols = sm.OLS(outcome, X).fit()

# Logit on any affair
logit = sm.Logit(any_affair, X).fit(disp=0)

# Output key results
print('Group summary (children=0 no, 1 yes):')
print(summary)
print('\nMean difference (yes - no):', yes_vals.mean() - no_vals.mean())
print('Welch t-test: t=%.3f p=%.4f' % (t_stat, p_val))
print('\nAny affair proportions:')
print('no=%.3f yes=%.3f' % (prop_no, prop_yes))
print('Prop z-test: z=%.3f p=%.4f' % (z, p_prop))

print('\nOLS children coef:', ols.params['children'], 'p=', ols.pvalues['children'])
print('Logit children coef:', logit.params['children'], 'p=', logit.pvalues['children'])

# For interpretation: compute marginal effect of children in logit (approx)
logit_margeff = logit.get_margeff(at='mean').summary_frame().loc['children']
print('Logit marginal effect (mean):', logit_margeff['dy/dx'])
