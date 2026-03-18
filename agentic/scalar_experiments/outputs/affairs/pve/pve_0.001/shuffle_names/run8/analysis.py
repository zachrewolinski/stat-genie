import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Map variables based on metadata descriptions
# 'religiousness' column is described as "Are there children in the marriage?" (yes/no)
# 'age' column is described as the affairs frequency
children = df['religiousness'].map({'yes': 1, 'no': 0})
affairs = df['age']

# Drop any missing
mask = children.notna() & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Group stats
mean_yes = affairs[children == 1].mean()
mean_no = affairs[children == 0].mean()
std_yes = affairs[children == 1].std(ddof=1)
std_no = affairs[children == 0].std(ddof=1)
n_yes = (children == 1).sum()
n_no = (children == 0).sum()

# Welch t-test
welch = stats.ttest_ind(affairs[children == 1], affairs[children == 0], equal_var=False, nan_policy='omit')

# Cohen's d (using pooled SD)
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression (affairs ~ children)
X = sm.add_constant(children)
model = sm.OLS(affairs, X).fit()

results = {
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'diff_yes_minus_no': float(mean_yes - mean_no),
    'welch_t': float(welch.statistic),
    'welch_p': float(welch.pvalue),
    'cohens_d': float(cohens_d),
    'ols_coef_children': float(model.params[1]),
    'ols_p_children': float(model.pvalues[1]),
}

print(results)
