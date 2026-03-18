import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Map columns
AFFAIR = 'feature2'
CHILDREN = 'feature6'

# Clean
# Ensure children is categorical with yes/no
_df = df[[AFFAIR, CHILDREN, 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']].copy()
_df = _df.dropna()

# Basic groups
children_yes = _df[_df[CHILDREN].str.lower() == 'yes'][AFFAIR]
children_no = _df[_df[CHILDREN].str.lower() == 'no'][AFFAIR]

# Summary stats
summary = {
    'n_yes': int(children_yes.shape[0]),
    'n_no': int(children_no.shape[0]),
    'mean_yes': float(children_yes.mean()),
    'mean_no': float(children_no.mean()),
    'median_yes': float(children_yes.median()),
    'median_no': float(children_no.median()),
    'prop_any_yes': float((children_yes > 0).mean()),
    'prop_any_no': float((children_no > 0).mean()),
}

# t-test (Welch)
t_stat, t_p = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Mann-Whitney U
u_stat, u_p = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

# Cohen's d
n1, n2 = children_yes.shape[0], children_no.shape[0]
var1, var2 = children_yes.var(ddof=1), children_no.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohens_d = (children_yes.mean() - children_no.mean()) / pooled_sd if pooled_sd > 0 else np.nan

# Proportion any affair: chi-square
contingency = pd.crosstab(_df[CHILDREN].str.lower(), _df[AFFAIR] > 0)
chi2, chi_p, _, _ = stats.chi2_contingency(contingency)

# Regression: affairs ~ children + controls
# Encode children as binary (yes=1)
_df['children_yes'] = (_df[CHILDREN].str.lower() == 'yes').astype(int)
# Encode gender
_df['male'] = (_df['feature3'].str.lower() == 'male').astype(int)

# OLS with controls
model = smf.ols(
    'feature2 ~ children_yes + male + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=_df
).fit()

results = {
    'summary': summary,
    't_test_p': float(t_p),
    't_test_stat': float(t_stat),
    'mannwhitney_p': float(u_p),
    'mannwhitney_stat': float(u_stat),
    'cohens_d': float(cohens_d),
    'chi2_p': float(chi_p),
    'chi2_stat': float(chi2),
    'reg_children_coef': float(model.params['children_yes']),
    'reg_children_p': float(model.pvalues['children_yes']),
}

print(json.dumps(results, indent=2))
