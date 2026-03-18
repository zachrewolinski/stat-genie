import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Map columns to meanings using descriptions in info.json
# children indicator is stored in column 'religiousness' (yes/no)
# affairs frequency is described under column 'age'

df['has_children'] = df['religiousness'].map({'yes': 1, 'no': 0})

# Outcome: engagement in extramarital affairs (affairs frequency)
affairs = df['age']

# Group summary
summary = df.groupby('has_children')['age'].agg(['count', 'mean', 'std'])

# t-test (Welch)
no_children = df.loc[df['has_children'] == 0, 'age']
children = df.loc[df['has_children'] == 1, 'age']

welch_t = stats.ttest_ind(children, no_children, equal_var=False)

# Cohen's d
mean_diff = children.mean() - no_children.mean()
pooled_sd = np.sqrt(((children.var(ddof=1) + no_children.var(ddof=1)) / 2))
cohens_d = mean_diff / pooled_sd

# Mann-Whitney U (two-sided)
mann = stats.mannwhitneyu(children, no_children, alternative='two-sided')

# OLS with controls (use known variables)
# Controls mapping:
# gender: 'gender'
# age (years): 'occupation'
# years married: 'children'
# education (years): 'yearsmarried'
# occupation class: 'rownames'
# religiousness (1-5): 'rating'
# marriage rating (1-5): 'affairs'

# Build dataframe with controls
model_df = df.copy()

# Ensure categorical
model_df['gender'] = model_df['gender'].astype('category')
model_df['has_children'] = model_df['has_children'].astype(int)

# OLS with robust SE
formula = 'age ~ has_children + C(gender) + occupation + children + yearsmarried + rownames + rating + affairs'
model = smf.ols(formula, data=model_df).fit(cov_type='HC3')

# Output results
print('Group summary (affairs frequency by children):')
print(summary)
print('\nWelch t-test:', welch_t)
print('Mean difference (children - no):', mean_diff)
print('Cohen d:', cohens_d)
print('Mann-Whitney U:', mann)
print('\nOLS with controls:')
print(model.summary())

