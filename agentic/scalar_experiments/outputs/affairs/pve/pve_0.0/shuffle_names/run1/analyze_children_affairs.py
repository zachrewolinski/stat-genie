import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# According to info.json metadata:
# - column 'religiousness' is children indicator (yes/no)
# - column 'age' is affairs frequency outcome

outcome = df['age']
children = df['religiousness'].map({'yes': 1, 'no': 0})

# Group stats
grp_yes = outcome[children == 1]
grp_no = outcome[children == 0]

mean_yes = grp_yes.mean()
mean_no = grp_no.mean()
std_yes = grp_yes.std(ddof=1)
std_no = grp_no.std(ddof=1)

# Welch t-test
t_stat, p_val = stats.ttest_ind(grp_yes, grp_no, equal_var=False)

# Cohen's d (using pooled SD)
n_yes = grp_yes.shape[0]
n_no = grp_no.shape[0]
pooled_sd = np.sqrt(((n_yes - 1) * std_yes ** 2 + (n_no - 1) * std_no ** 2) / (n_yes + n_no - 2))
d = (mean_yes - mean_no) / pooled_sd

# Adjusted regression controlling for other columns (excluding the ID-like 'education')
# Treat gender as binary, include other numeric predictors.
X = pd.DataFrame({
    'children_yes': children,
    'gender_male': (df['gender'] == 'male').astype(int),
    'age_category': df['occupation'],
    'years_married_code': df['children'],
    'religiosity_code': df['rating'],
    'marriage_rating': df['affairs'],
    'education_code': df['yearsmarried'],
    'occupation_code': df['rownames'],
})
X = sm.add_constant(X)
model = sm.OLS(outcome, X, missing='drop').fit()

# Extract adjusted effect for children
coef = model.params['children_yes']
p_adj = model.pvalues['children_yes']

# Output summary stats for manual write-up
print('n_yes', n_yes, 'n_no', n_no)
print('mean_yes', mean_yes, 'mean_no', mean_no)
print('std_yes', std_yes, 'std_no', std_no)
print('t_stat', t_stat, 'p_val', p_val)
print('cohens_d', d)
print('adj_coef', coef, 'adj_p', p_adj)
