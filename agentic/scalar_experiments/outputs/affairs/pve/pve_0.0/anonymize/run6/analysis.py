import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map children yes/no
_df['has_children'] = _df['feature6'].str.lower().map({'yes': 1, 'no': 0})

# Outcome
_y = _df['feature2']

# Group stats
summary = _df.groupby('has_children')['feature2'].agg(['count', 'mean', 'std', 'median'])

# Welch t-test
no_group = _df[_df['has_children'] == 0]['feature2']
yes_group = _df[_df['has_children'] == 1]['feature2']

t_stat, t_p = stats.ttest_ind(yes_group, no_group, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use alternative='two-sided' (default) and handle ties with scipy
u_stat, u_p = stats.mannwhitneyu(yes_group, no_group, alternative='two-sided')

# Effect size (Cohen's d, using pooled SD)
mean_diff = yes_group.mean() - no_group.mean()
pooled_sd = np.sqrt(((yes_group.var(ddof=1) + no_group.var(ddof=1)) / 2))
cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression: feature2 ~ has_children
X = sm.add_constant(_df['has_children'])
ols = sm.OLS(_df['feature2'], X).fit()

results = {
    'summary': summary.to_dict(),
    'mean_diff_yes_minus_no': float(mean_diff),
    'cohens_d': float(cohens_d),
    't_stat': float(t_stat),
    't_p': float(t_p),
    'u_stat': float(u_stat),
    'u_p': float(u_p),
    'ols_coef_has_children': float(ols.params['has_children']),
    'ols_pvalue_has_children': float(ols.pvalues['has_children']),
    'ols_r2': float(ols.rsquared),
}

print(json.dumps(results, indent=2))
