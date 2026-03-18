import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Map shuffled columns to semantic variables using info.json descriptions
# children indicator is stored in column 'religiousness' (values: yes/no)
# affairs frequency is stored in column 'age' per metadata description

affairs = df['age']
children = df['religiousness'].map({'yes': 1, 'no': 0})

# Basic group statistics
stats_by_child = df.groupby('religiousness')['age'].agg(['count', 'mean', 'std'])

# Two-sample t-test (Welch)
no_affairs = df.loc[df['religiousness'] == 'no', 'age']
yes_affairs = df.loc[df['religiousness'] == 'yes', 'age']

t_stat, p_value = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False, nan_policy='omit')

# Effect size: Cohen's d
n1, n0 = len(yes_affairs), len(no_affairs)
var1, var0 = np.nanvar(yes_affairs, ddof=1), np.nanvar(no_affairs, ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohens_d = (yes_affairs.mean() - no_affairs.mean()) / pooled_sd

# Regression with controls (using mapped variables from metadata)
# age_years: occupation, years_married: children, religiousness_level: rating
# education_level: yearsmarried, occupation: rownames, marriage_rating: affairs
# gender encoded as binary
controls = pd.DataFrame({
    'children': children,
    'age_years': df['occupation'],
    'years_married': df['children'],
    'religiousness_level': df['rating'],
    'education_level': df['yearsmarried'],
    'occupation_code': df['rownames'],
    'marriage_rating': df['affairs'],
    'gender_male': (df['gender'] == 'male').astype(int),
})

X = sm.add_constant(controls)
model = sm.OLS(affairs, X, missing='drop').fit()

print('Group stats (affairs by children):')
print(stats_by_child)
print('\nWelch t-test: t=%.4f, p=%.6f' % (t_stat, p_value))
print('Cohen d: %.4f' % cohens_d)
print('\nRegression coefficient for children:')
print(model.params['children'], model.pvalues['children'])

