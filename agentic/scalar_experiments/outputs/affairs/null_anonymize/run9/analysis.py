import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# Rename columns for clarity based on info.json
rename = {
    'feature1': 'id',
    'feature2': 'affairs',
    'feature3': 'gender',
    'feature4': 'age',
    'feature5': 'years_married',
    'feature6': 'children',
    'feature7': 'religiousness',
    'feature8': 'education',
    'feature9': 'occupation',
    'feature10': 'marriage_rating',
}

df = df.rename(columns=rename)

# Basic counts by children
print(df['children'].value_counts())

# Affair engagement: binary (any affair)
df['affair_any'] = df['affairs'] > 0

summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any=('affair_any', 'mean'),
)
print(summary)

# Effect sizes
if set(df['children'].unique()) >= {'yes', 'no'}:
    mean_diff = summary.loc['yes', 'mean_affairs'] - summary.loc['no', 'mean_affairs']
    prop_diff = summary.loc['yes', 'prop_any'] - summary.loc['no', 'prop_any']
    print('mean_diff yes-no', mean_diff)
    print('prop_diff yes-no', prop_diff)

# Statistical tests: t-test for mean, chi-square for proportions

yes_aff = df.loc[df['children'] == 'yes', 'affairs']
no_aff = df.loc[df['children'] == 'no', 'affairs']
print('t-test', stats.ttest_ind(yes_aff, no_aff, equal_var=False))

# chi-square for any affair
cont = pd.crosstab(df['children'], df['affair_any'])
print('contingency\n', cont)
chi2, p, dof, exp = stats.chi2_contingency(cont)
print('chi2', chi2, 'p', p)

# logistic regression controlling for covariates
# encode children yes=1 no=0

df['children_yes'] = (df['children'] == 'yes').astype(int)

# simple logistic regression for affair_any
X = df[['children_yes', 'age', 'years_married', 'religiousness', 'education', 'occupation', 'marriage_rating']]
X = sm.add_constant(X)
model = sm.Logit(df['affair_any'].astype(int), X).fit(disp=False)
print(model.summary())

# coefficient for children
print('coef children', model.params['children_yes'], 'p', model.pvalues['children_yes'])

# Also linear regression on affairs
model_lin = sm.OLS(df['affairs'], X).fit()
print(model_lin.summary())
print('lin coef children', model_lin.params['children_yes'], 'p', model_lin.pvalues['children_yes'])
