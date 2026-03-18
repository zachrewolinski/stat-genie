import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Map variables based on info.json descriptions
# children indicator is in column 'religiousness' (yes/no)
# affairs engagement is in column 'age' per metadata description

df['has_children'] = (df['religiousness'].str.lower() == 'yes').astype(int)

affairs = df['age']

# Group statistics
mean_yes = affairs[df['has_children'] == 1].mean()
mean_no = affairs[df['has_children'] == 0].mean()
std_yes = affairs[df['has_children'] == 1].std(ddof=1)
std_no = affairs[df['has_children'] == 0].std(ddof=1)

n_yes = (df['has_children'] == 1).sum()
n_no = (df['has_children'] == 0).sum()

# Welch t-test
welch = stats.ttest_ind(
    affairs[df['has_children'] == 1],
    affairs[df['has_children'] == 0],
    equal_var=False,
    nan_policy='omit'
)

# Effect size: Cohen's d (using pooled SD)
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Nonparametric Mann-Whitney U
mw = stats.mannwhitneyu(
    affairs[df['has_children'] == 1],
    affairs[df['has_children'] == 0],
    alternative='two-sided'
)

# Regression with controls based on metadata mapping
# age categories -> column 'occupation'
# years married -> column 'children'
# education level -> column 'yearsmarried'
# occupation category -> column 'rownames'
# religiousness scale -> column 'rating'
# marriage rating -> column 'affairs'
# gender -> column 'gender'

model = smf.ols(
    'age ~ has_children + C(gender) + occupation + children + yearsmarried + rating + affairs + rownames',
    data=df
).fit(cov_type='HC3')

result = {
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'diff_mean': float(mean_yes - mean_no),
    'welch_t': float(welch.statistic),
    'welch_p': float(welch.pvalue),
    'cohens_d': float(cohens_d),
    'mw_u': float(mw.statistic),
    'mw_p': float(mw.pvalue),
    'reg_coef': float(model.params['has_children']),
    'reg_p': float(model.pvalues['has_children']),
}

print(json.dumps(result, indent=2))
