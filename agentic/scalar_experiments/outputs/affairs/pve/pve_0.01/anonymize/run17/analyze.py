import json
import pandas as pd
import numpy as np
from scipy import stats

DATA_PATH = 'affairs.csv'

df = pd.read_csv(DATA_PATH)

# Identify columns
# feature2: number of affairs frequency (numeric)
# feature6: children yes/no

# Clean/standardize
children = df['feature6'].astype(str).str.lower().str.strip()

# Map to boolean
has_children = children == 'yes'

# Outcome: affair frequency (numeric)
affairs = df['feature2']

# Basic group stats
summary = df.groupby(has_children)['feature2'].agg(['count','mean','median','std'])

# Two-sample t-test (Welch) for mean difference
x = affairs[has_children]
y = affairs[~has_children]

t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d, using pooled SD)
# Handle possible zero-variance
nx, ny = len(x), len(y)
var_x = np.nanvar(x, ddof=1)
var_y = np.nanvar(y, ddof=1)
pooled_sd = np.sqrt(((nx-1)*var_x + (ny-1)*var_y) / (nx + ny - 2)) if (nx+ny-2) > 0 else np.nan
cohens_d = (np.nanmean(x) - np.nanmean(y)) / pooled_sd if pooled_sd and pooled_sd>0 else np.nan

# Nonparametric test (Mann-Whitney U)
try:
    u_stat, u_p = stats.mannwhitneyu(x, y, alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

# Also check proportion with any affair >0
any_affair = affairs > 0
prop_children = any_affair[has_children].mean()
prop_no_children = any_affair[~has_children].mean()

# Chi-square test for proportions
contingency = pd.crosstab(has_children, any_affair)
chi2, chi2_p, dof, exp = stats.chi2_contingency(contingency)

results = {
    'summary': summary.to_dict(),
    't_test': {'t_stat': t_stat, 'p_val': p_val},
    'cohens_d': cohens_d,
    'mannwhitney': {'u_stat': u_stat, 'p_val': u_p},
    'prop_any_affair': {'children': prop_children, 'no_children': prop_no_children},
    'chi2': {'chi2': chi2, 'p_val': chi2_p, 'dof': dof},
    'counts': {'children_yes': int(has_children.sum()), 'children_no': int((~has_children).sum())}
}

print(json.dumps(results, indent=2))
