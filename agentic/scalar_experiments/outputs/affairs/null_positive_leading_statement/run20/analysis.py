import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic cleaning
df['children_yes'] = (df['children'].str.lower() == 'yes').astype(int)
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group stats
group = df.groupby('children_yes')
mean_affairs = group['affairs'].mean()
prop_any = group['any_affair'].mean()
n = group.size()

# t-test on affairs counts
yes = df[df['children_yes'] == 1]['affairs']
no = df[df['children_yes'] == 0]['affairs']
t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')

# Logistic regression for any affair
logit_simple = smf.logit('any_affair ~ children_yes', data=df).fit(disp=False)

# Logistic regression with controls
# Use C(gender) for categorical
logit_controls = smf.logit(
    'any_affair ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df
).fit(disp=False)

# Poisson regression for count affairs
poisson_simple = smf.glm('affairs ~ children_yes', data=df, family=sm.families.Poisson()).fit()
poisson_controls = smf.glm(
    'affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df,
    family=sm.families.Poisson()
).fit()

# Effect sizes
mean_diff = mean_affairs.loc[1] - mean_affairs.loc[0]
prop_diff = prop_any.loc[1] - prop_any.loc[0]

print('N yes/no:', n.to_dict())
print('Mean affairs (no children=0, yes children=1):', mean_affairs.to_dict())
print('Mean diff (yes - no):', mean_diff)
print('Prop any affair (no children=0, yes children=1):', prop_any.to_dict())
print('Prop diff (yes - no):', prop_diff)
print('t-test affairs: t=%.3f p=%.4f' % (t_stat, t_p))
print('Mann-Whitney U p=%.4f' % u_p)

# Logit results
print('\nLogit simple children_yes coef:', logit_simple.params['children_yes'], 'p=', logit_simple.pvalues['children_yes'])
print('Logit controls children_yes coef:', logit_controls.params['children_yes'], 'p=', logit_controls.pvalues['children_yes'])

# Poisson results
print('\nPoisson simple children_yes coef:', poisson_simple.params['children_yes'], 'p=', poisson_simple.pvalues['children_yes'])
print('Poisson controls children_yes coef:', poisson_controls.params['children_yes'], 'p=', poisson_controls.pvalues['children_yes'])

# Odds ratio and IRR
print('\nLogit controls OR:', np.exp(logit_controls.params['children_yes']))
print('Poisson controls IRR:', np.exp(poisson_controls.params['children_yes']))
