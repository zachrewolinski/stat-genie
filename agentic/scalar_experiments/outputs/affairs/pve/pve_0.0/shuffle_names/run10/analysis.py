import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map variables based on info.json descriptions
# 'religiousness' description indicates children yes/no
# 'age' description indicates extramarital affairs frequency

children = _df['religiousness'].map({'yes': 1, 'no': 0})
affairs = _df['age']

# Drop missing
mask = children.notna() & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Group statistics
mean_yes = affairs[children == 1].mean()
mean_no = affairs[children == 0].mean()
std_yes = affairs[children == 1].std(ddof=1)
std_no = affairs[children == 0].std(ddof=1)

n_yes = (children == 1).sum()
n_no = (children == 0).sum()

# Welch t-test
res_t = stats.ttest_ind(affairs[children==1], affairs[children==0], equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
res_u = stats.mannwhitneyu(affairs[children==1], affairs[children==0], alternative='two-sided')

# Effect size (Cohen's d) using pooled std (unequal n)
# Use pooled std with ddof=1
pooled_var = ((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2)
cohens_d = (mean_yes - mean_no) / np.sqrt(pooled_var)

# Regression with controls (use other columns as covariates; encode categorical)
# Use 'gender' and 'rating' (religiousness) and others as covariates if numeric
# Keep it simple: affairs ~ children + gender + yearsmarried + rating + education + occupation + rownames + affairs? no

# Build design matrix
X = pd.DataFrame({
    'children': children,
    'gender_male': (_df.loc[mask, 'gender'] == 'male').astype(int),
    'yearsmarried': _df.loc[mask, 'children'],  # description indicates yearsmarried
    'education': _df.loc[mask, 'yearsmarried'],  # description indicates education
    'occupation': _df.loc[mask, 'rownames'],     # description indicates occupation
    'religiousness': _df.loc[mask, 'rating'],    # description indicates religiousness
    'rating': _df.loc[mask, 'affairs'],          # description indicates rating of marriage
})

X = sm.add_constant(X)
model = sm.OLS(affairs, X).fit()

results = {
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'std_yes': float(std_yes),
    'std_no': float(std_no),
    't_stat': float(res_t.statistic),
    't_pvalue': float(res_t.pvalue),
    'u_stat': float(res_u.statistic),
    'u_pvalue': float(res_u.pvalue),
    'cohens_d': float(cohens_d),
    'reg_children_coef': float(model.params['children']),
    'reg_children_pvalue': float(model.pvalues['children']),
}

print(json.dumps(results, indent=2))
