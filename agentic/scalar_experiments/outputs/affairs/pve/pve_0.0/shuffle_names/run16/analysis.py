import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Map columns based on metadata descriptions
# Outcome: extramarital affairs frequency (metadata description attached to column named 'age')
# Predictor: children present (metadata description attached to column named 'religiousness')
model_df = pd.DataFrame({
    'affairs_score': df['age'],
    'children_present': df['religiousness'].map({'yes': 1, 'no': 0}),
    'years_married': df['children'],
    'age_cat': df['occupation'],
    'education_years': df['yearsmarried'],
    'religiousness': df['rating'],
    'occupation': df['rownames'],
    'marriage_rating': df['affairs'],
    'gender': df['gender'],
})

# Drop missing (should be none)
model_df = model_df.dropna()

# Group stats
by_child = model_df.groupby('children_present')['affairs_score']
mean_no = by_child.mean().loc[0]
mean_yes = by_child.mean().loc[1]
std_no = by_child.std(ddof=1).loc[0]
std_yes = by_child.std(ddof=1).loc[1]
n_no = by_child.size().loc[0]
n_yes = by_child.size().loc[1]

# Difference (yes - no)
diff = mean_yes - mean_no

# Welch t-test
welch = stats.ttest_ind(
    model_df.loc[model_df['children_present'] == 1, 'affairs_score'],
    model_df.loc[model_df['children_present'] == 0, 'affairs_score'],
    equal_var=False
)

# Mann-Whitney U
mwu = stats.mannwhitneyu(
    model_df.loc[model_df['children_present'] == 1, 'affairs_score'],
    model_df.loc[model_df['children_present'] == 0, 'affairs_score'],
    alternative='two-sided'
)

# Cohen's d (using pooled SD)
pooled_sd = np.sqrt(((n_no - 1) * std_no**2 + (n_yes - 1) * std_yes**2) / (n_no + n_yes - 2))
cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan

# OLS unadjusted
ols_simple = smf.ols('affairs_score ~ children_present', data=model_df).fit()

# OLS adjusted for other covariates
ols_adj = smf.ols(
    'affairs_score ~ children_present + years_married + age_cat + education_years + religiousness + occupation + marriage_rating + C(gender)',
    data=model_df
).fit()

print('Group sizes (no, yes):', n_no, n_yes)
print('Means (no, yes):', mean_no, mean_yes)
print('Diff (yes - no):', diff)
print('Welch t-test:', welch)
print('Mann-Whitney U:', mwu)
print("Cohen's d:", cohens_d)
print('\nOLS simple:\n', ols_simple.summary())
print('\nOLS adjusted:\n', ols_adj.summary())
