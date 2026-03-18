import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# Based on metadata descriptions:
# - column 'religiousness' indicates whether there are children in the marriage (yes/no)
# - column 'age' corresponds to engagement in extramarital affairs (frequency scale)
children = df['religiousness'].map({'yes': 1, 'no': 0})
affairs = df['age']

# Basic group stats
summary = df.groupby('religiousness')['age'].agg(['count','mean','std'])
print('Group summary (affairs by children yes/no):')
print(summary)

# Welch t-test
x_yes = affairs[children==1]
x_no = affairs[children==0]

t_stat, p_val = stats.ttest_ind(x_yes, x_no, equal_var=False)
print('\nWelch t-test:')
print('t =', t_stat, 'p =', p_val)

# Mann-Whitney U test
u_stat, p_u = stats.mannwhitneyu(x_yes, x_no, alternative='two-sided')
print('\nMann-Whitney U:')
print('U =', u_stat, 'p =', p_u)

# Effect size (Cohen's d)
mean_diff = x_yes.mean() - x_no.mean()
pooled_sd = np.sqrt(((x_yes.std(ddof=1)**2) + (x_no.std(ddof=1)**2)) / 2)
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
print('\nEffect size:')
print('mean diff (yes - no) =', mean_diff)
print('Cohen d =', cohen_d)

# Simple OLS with controls from available variables (as per descriptions)
# Controls: age categories (occupation), years married (children column),
# religiosity (rating), education (yearsmarried), occupation (rownames),
# marriage rating (affairs column), gender.

# Build design matrix
controls = pd.DataFrame({
    'children_yes': children,
    'gender_male': (df['gender'] == 'male').astype(int),
    'age_cat': df['occupation'],
    'years_married': df['children'],
    'religiousness_scale': df['rating'],
    'education_level': df['yearsmarried'],
    'occupation_level': df['rownames'],
    'marriage_rating': df['affairs'],
})

X = sm.add_constant(controls)
model = sm.OLS(affairs, X).fit()
print('\nOLS (affairs ~ children + controls) coef/p-value for children:')
print('coef =', model.params['children_yes'], 'p =', model.pvalues['children_yes'])

