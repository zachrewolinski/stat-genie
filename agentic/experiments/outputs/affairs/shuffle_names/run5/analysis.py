import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map shuffled columns back to their likely meaning by value ranges
# Affairs frequency is coded as 0,1,2,3,7,12 -> stored in column named 'age'
# Children is yes/no -> stored in column named 'religiousness'
# Age categories 17.5..57 -> stored in column named 'occupation'
# Years married 0.125..15 -> stored in column named 'children'
# Education 9..20 -> stored in column named 'yearsmarried'
# Occupation 1..7 -> stored in column named 'rownames'
# Remaining two 1..5 scales are religiousness and marriage rating
mapped = pd.DataFrame({
    'affairs': _df['age'],
    'children': (_df['religiousness'] == 'yes').astype(int),
    'age': _df['occupation'],
    'years_married': _df['children'],
    'education': _df['yearsmarried'],
    'occupation': _df['rownames'],
    'gender': _df['gender'],
    'religiousness': _df['rating'],
    'rate_marriage': _df['affairs'],
})

# Basic checks to guard against unexpected schema
assert mapped['affairs'].isin([0, 1, 2, 3, 7, 12]).all(), 'Affairs coding unexpected'
assert mapped['children'].isin([0, 1]).all(), 'Children should be binary'

# Descriptives: mean affairs by children status
mean_by_children = mapped.groupby('children')['affairs'].mean()
count_by_children = mapped.groupby('children')['affairs'].size()

# Two-sample t-test (Welch)
_yes = mapped.loc[mapped['children'] == 1, 'affairs']
_no = mapped.loc[mapped['children'] == 0, 'affairs']
welch_t = stats.ttest_ind(_yes, _no, equal_var=False)

# Regression with controls (Poisson for count outcome)
formula = (
    'affairs ~ children + age + years_married + education + occupation '
    '+ C(gender) + religiousness + rate_marriage'
)
pois = smf.glm(formula, data=mapped, family=sm.families.Poisson()).fit()

print('Counts by children status')
print(count_by_children)
print('\nMean affairs by children status (0=no, 1=yes)')
print(mean_by_children)
print('\nWelch t-test (yes vs no):')
print(welch_t)
print('\nPoisson regression (coefficients):')
print(pois.summary().tables[1])
