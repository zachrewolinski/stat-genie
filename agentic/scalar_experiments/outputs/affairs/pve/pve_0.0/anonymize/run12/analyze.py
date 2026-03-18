import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

print(df.head())
print(df.dtypes)

# Basic group stats
for col in ['feature2']:
    group_stats = df.groupby('feature6')[col].agg(['count', 'mean', 'median', 'std'])
    print('Group stats for', col)
    print(group_stats)

# t-test for difference in means
children_yes = df[df['feature6'] == 'yes']['feature2']
children_no = df[df['feature6'] == 'no']['feature2']
# Welch t-test

t_res = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')
print('Welch t-test feature2 by children yes/no:', t_res)

# Nonparametric test (Mann-Whitney)

mw_res = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
print('Mann-Whitney U:', mw_res)

# Analyze binary: any affair >0

df['any_affair'] = (df['feature2'] > 0).astype(int)

# Proportion by children

prop = df.groupby('feature6')['any_affair'].mean()
counts = df.groupby('feature6')['any_affair'].agg(['sum', 'count'])
print('Any affair proportion by children')
print(prop)
print(counts)

# Chi-square test for independence

contingency = pd.crosstab(df['feature6'], df['any_affair'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print('Chi-square:', chi2, 'p=', p)
print('contingency')
print(contingency)

# Logistic regression: any_affair ~ children
# encode children: yes=1, no=0

df['children_yes'] = (df['feature6'] == 'yes').astype(int)
X = sm.add_constant(df['children_yes'])
model = sm.Logit(df['any_affair'], X).fit(disp=False)
print(model.summary())

# OLS on feature2 (affair count) ~ children

ols = sm.OLS(df['feature2'], X).fit()
print(ols.summary())
