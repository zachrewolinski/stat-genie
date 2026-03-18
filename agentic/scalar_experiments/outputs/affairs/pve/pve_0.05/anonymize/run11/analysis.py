import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Rename columns for clarity

df = df.rename(columns={
    'feature2': 'affairs',
    'feature6': 'children',
    'feature3': 'gender',
    'feature4': 'age',
    'feature5': 'years_married',
    'feature7': 'religiousness',
    'feature8': 'education',
    'feature9': 'occupation',
    'feature10': 'marriage_rating'
})

# Basic group stats

df['children'] = df['children'].str.lower()

groups = df.groupby('children')['affairs']

stats_summary = groups.agg(['count', 'mean', 'median', 'std'])
nonzero_rate = groups.apply(lambda s: (s > 0).mean())

# Welch t-test for mean differences

yes = df.loc[df['children'] == 'yes', 'affairs']
no = df.loc[df['children'] == 'no', 'affairs']

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparam)

u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')

# Effect size (Cohen's d)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = ((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)
    return (np.mean(a) - np.mean(b)) / np.sqrt(s_pooled)


d = cohens_d(yes, no)

# Logistic regression for any affair

df['any_affair'] = (df['affairs'] > 0).astype(int)
logit = smf.logit('any_affair ~ C(children)', data=df).fit(disp=False)

# OLS for affairs (simple)

ols = smf.ols('affairs ~ C(children)', data=df).fit()

# OLS with controls

ols_ctrl = smf.ols(
    'affairs ~ C(children) + age + years_married + C(gender) + religiousness + education + occupation + marriage_rating',
    data=df
).fit()

# Logistic with controls

logit_ctrl = smf.logit(
    'any_affair ~ C(children) + age + years_married + C(gender) + religiousness + education + occupation + marriage_rating',
    data=df
).fit(disp=False)

# Print summary results

print('group_stats')
print(stats_summary)
print('nonzero_rate')
print(nonzero_rate)
print('welch_t', t_stat, t_p)
print('mannwhitney', u_stat, u_p)
print('cohens_d', d)

print('ols_children_coef', ols.params.get('C(children)[T.yes]'), ols.pvalues.get('C(children)[T.yes]'))
print('ols_ctrl_children_coef', ols_ctrl.params.get('C(children)[T.yes]'), ols_ctrl.pvalues.get('C(children)[T.yes]'))

print('logit_children_coef', logit.params.get('C(children)[T.yes]'), logit.pvalues.get('C(children)[T.yes]'))
print('logit_ctrl_children_coef', logit_ctrl.params.get('C(children)[T.yes]'), logit_ctrl.pvalues.get('C(children)[T.yes]'))
