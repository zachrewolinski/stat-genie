import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data and metadata
with open('info.json', 'r') as f:
    info = json.load(f)

# Dataset
_df = pd.read_csv('affairs.csv')

# Identify children indicator column and affairs outcome based on metadata
# The metadata description for column 'religiousness' says: "Are there children in the marriage?"
# The metadata description for column 'age' says it is the frequency of extramarital intercourse.
children_col = 'religiousness'
outcome_col = 'age'

# Map children to binary
children = _df[children_col].map({'yes': 1, 'no': 0})

# Outcome
outcome = _df[outcome_col]

# Drop missing if any
mask = children.notna() & outcome.notna()
children = children[mask]
outcome = outcome[mask]

# Group stats
mean_yes = outcome[children == 1].mean()
mean_no = outcome[children == 0].mean()
std_yes = outcome[children == 1].std(ddof=1)
std_no = outcome[children == 0].std(ddof=1)

# Two-sample t-test (Welch)
t_stat, p_val = stats.ttest_ind(
    outcome[children == 1], outcome[children == 0], equal_var=False, nan_policy='omit'
)

# Effect size (Cohen's d using pooled SD)
pooled_sd = np.sqrt(((std_yes ** 2) + (std_no ** 2)) / 2)
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Mann-Whitney U test (non-parametric)
try:
    u_stat, p_u = stats.mannwhitneyu(
        outcome[children == 1], outcome[children == 0], alternative='two-sided'
    )
except ValueError:
    p_u = np.nan

# Simple regression: outcome ~ children
X = sm.add_constant(children.astype(float))
model = sm.OLS(outcome.values, X.values).fit()

# Regression controlling for other numeric covariates (excluding outcome and children)
# Use numeric columns only, avoid constant column
numeric_cols = [c for c in _df.columns if _df[c].dtype != 'object' and c not in {outcome_col}]
X2 = _df[numeric_cols].copy()
X2['children'] = children
X2 = sm.add_constant(X2)
X2 = X2.loc[mask]
model2 = sm.OLS(outcome.loc[mask].values, X2.values).fit()

results = {
    'n': int(mask.sum()),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'diff_yes_minus_no': float(mean_yes - mean_no),
    't_stat': float(t_stat),
    'p_val': float(p_val),
    'cohens_d': float(cohens_d),
    'p_u': float(p_u),
    'reg_coef_children': float(model.params[1]),
    'reg_pval_children': float(model.pvalues[1]),
    'reg2_coef_children': float(model2.params[list(X2.columns).index('children')]),
    'reg2_pval_children': float(model2.pvalues[list(X2.columns).index('children')]),
}

print(json.dumps(results, indent=2))
