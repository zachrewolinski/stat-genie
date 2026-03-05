import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns based on info.json
# feature2 = affairs frequency
# feature6 = children yes/no

# Clean
if _df['feature6'].dtype != 'object':
    _df['feature6'] = _df['feature6'].astype(str)

# Normalize children values
_df['feature6'] = _df['feature6'].str.strip().str.lower()

# Binary children indicator
_df['children_yes'] = (_df['feature6'] == 'yes').astype(int)

# Affairs frequency (numeric)
affairs = pd.to_numeric(_df['feature2'], errors='coerce')

# Any affair
_df['any_affair'] = (affairs > 0).astype(int)

# Drop missing
_df = _df.dropna(subset=['children_yes', 'feature2'])

# Group stats
stats_by_child = _df.groupby('children_yes')['feature2'].agg(['count', 'mean', 'median', 'std'])

# Proportion with any affair
prop_any = _df.groupby('children_yes')['any_affair'].mean()

# Mann-Whitney U test (non-parametric) for distribution difference
with_children = _df.loc[_df['children_yes'] == 1, 'feature2']
without_children = _df.loc[_df['children_yes'] == 0, 'feature2']

u_stat, u_p = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')

# Cliff's delta effect size
# Compute delta: (number of pairs where x>y - x<y) / (n1*n2)
# Use efficient computation via ranking if sizes large.

# Direct computation for moderate sizes
x = with_children.to_numpy()
y = without_children.to_numpy()

# To reduce memory/time, use broadcasting in chunks
n1, n2 = len(x), len(y)

# If very large, chunk; here sizes are small (<=601)
count_greater = 0
count_less = 0
for xi in x:
    count_greater += np.sum(xi > y)
    count_less += np.sum(xi < y)

cliffs_delta = (count_greater - count_less) / (n1 * n2)

# Chi-square test for any affair vs children
contingency = pd.crosstab(_df['children_yes'], _df['any_affair'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

# Difference in proportions
prop_diff = prop_any.loc[1] - prop_any.loc[0]

# Simple logistic regression (unadjusted) using statsmodels if available
try:
    import statsmodels.api as sm
    X = sm.add_constant(_df['children_yes'])
    y_bin = _df['any_affair']
    logit_model = sm.Logit(y_bin, X)
    logit_res = logit_model.fit(disp=False)
    logit_p = float(logit_res.pvalues['children_yes'])
    logit_coef = float(logit_res.params['children_yes'])
except Exception:
    logit_p = None
    logit_coef = None

output = {
    'n_total': int(len(_df)),
    'n_children_yes': int(( _df['children_yes'] == 1).sum()),
    'n_children_no': int(( _df['children_yes'] == 0).sum()),
    'mean_affairs_children_yes': float(stats_by_child.loc[1, 'mean']),
    'mean_affairs_children_no': float(stats_by_child.loc[0, 'mean']),
    'median_affairs_children_yes': float(stats_by_child.loc[1, 'median']),
    'median_affairs_children_no': float(stats_by_child.loc[0, 'median']),
    'prop_any_affair_children_yes': float(prop_any.loc[1]),
    'prop_any_affair_children_no': float(prop_any.loc[0]),
    'prop_any_diff_yes_minus_no': float(prop_diff),
    'mannwhitney_u_stat': float(u_stat),
    'mannwhitney_p': float(u_p),
    'cliffs_delta': float(cliffs_delta),
    'chi2_stat': float(chi2),
    'chi2_p': float(chi_p),
    'logit_coef_children_yes': logit_coef,
    'logit_p_children_yes': logit_p,
}

with open('analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
