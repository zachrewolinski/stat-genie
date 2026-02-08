import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Map columns based on metadata mismatch
# children indicator is in 'religiousness' column (yes/no)
# affairs frequency is in 'age' column (0,1,2,3,7,12)

affairs = df['age']
children = df['religiousness'].map({'yes':1, 'no':0})

# Group stats
means = df.groupby('religiousness')['age'].mean()
medians = df.groupby('religiousness')['age'].median()
counts = df['religiousness'].value_counts()

# t-test
age_yes = df.loc[df['religiousness']=='yes','age']
age_no = df.loc[df['religiousness']=='no','age']

t_stat, t_p = stats.ttest_ind(age_yes, age_no, equal_var=False)

# Mann-Whitney
u_stat, u_p = stats.mannwhitneyu(age_yes, age_no, alternative='two-sided')

# Simple regression (OLS) with controls based on likely mappings
# Use affairs frequency as dependent
# Controls: gender, yearsmarried (actually education), rating (religiousness), occupation (age category), rownames (occupation), children(yes/no)

X = pd.DataFrame({
    'children': children,
    'gender_male': (df['gender']=='male').astype(int),
    'education_years': df['yearsmarried'],  # mislabeled
    'religiousness_level': df['rating'],    # mislabeled
    'age_category': df['occupation'],
    'occupation_class': df['rownames'],
    'years_married': df['children'],
    'marriage_rating': df['affairs'],
})
X = sm.add_constant(X)

model = sm.OLS(affairs, X, missing='drop').fit()

print('Group means (affairs frequency):')
print(means)
print('Group medians:')
print(medians)
print('Counts:')
print(counts)
print('\nT-test: t=%.3f p=%.4g' % (t_stat, t_p))
print('Mann-Whitney U: U=%.1f p=%.4g' % (u_stat, u_p))

coef = model.params['children']
ci_low, ci_high = model.conf_int().loc['children']
print('\nOLS children coef:', coef)
print('95% CI:', (ci_low, ci_high))
print('p-value:', model.pvalues['children'])
