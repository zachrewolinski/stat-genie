import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning
# Expect feature6 to indicate children yes/no
_df = _df.copy()
_df['children'] = _df['feature6'].astype(str).str.lower().str.strip()
_df = _df[_df['children'].isin(['yes', 'no'])]
_df['children_bin'] = (_df['children'] == 'yes').astype(int)

# Outcome
_y = _df['feature2'].astype(float)

# Group summaries
summary = _df.groupby('children').agg(
    n=('feature2', 'size'),
    mean=('feature2', 'mean'),
    median=('feature2', 'median'),
    std=('feature2', 'std'),
    zero_rate=('feature2', lambda x: np.mean(np.isclose(x, 0)))
).reset_index()

# t-test (Welch) children yes vs no
vals_yes = _df.loc[_df['children']=='yes', 'feature2'].astype(float)
vals_no = _df.loc[_df['children']=='no', 'feature2'].astype(float)

t_stat, t_p = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d (yes vs no)
# d = (mean_yes - mean_no) / pooled_std
mean_yes = vals_yes.mean()
mean_no = vals_no.mean()
std_yes = vals_yes.std(ddof=1)
std_no = vals_no.std(ddof=1)
pooled_std = np.sqrt(((std_yes**2) + (std_no**2)) / 2)
cohen_d = (mean_yes - mean_no) / pooled_std if pooled_std > 0 else np.nan

# Simple OLS: feature2 ~ children
ols_simple = smf.ols('feature2 ~ children_bin', data=_df).fit(cov_type='HC3')

# Adjusted OLS with covariates where available
# Use feature3 as categorical, others numeric
formula = 'feature2 ~ children_bin + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
ols_adj = smf.ols(formula, data=_df).fit(cov_type='HC3')

results = {
    'summary': summary.to_dict(orient='records'),
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mann_whitney': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    'effect': {
        'mean_yes': float(mean_yes),
        'mean_no': float(mean_no),
        'mean_diff_yes_minus_no': float(mean_yes - mean_no),
        'cohen_d': float(cohen_d)
    },
    'ols_simple': {
        'coef_children': float(ols_simple.params['children_bin']),
        'p_value_children': float(ols_simple.pvalues['children_bin']),
        'ci_low': float(ols_simple.conf_int().loc['children_bin', 0]),
        'ci_high': float(ols_simple.conf_int().loc['children_bin', 1])
    },
    'ols_adj': {
        'coef_children': float(ols_adj.params['children_bin']),
        'p_value_children': float(ols_adj.pvalues['children_bin']),
        'ci_low': float(ols_adj.conf_int().loc['children_bin', 0]),
        'ci_high': float(ols_adj.conf_int().loc['children_bin', 1])
    }
}

print(json.dumps(results, indent=2))
