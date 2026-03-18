import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns for readability
col_map = {
    'feature2': 'affairs',  # frequency
    'feature6': 'children',
}

# Ensure expected columns
for c in col_map:
    if c not in _df.columns:
        raise KeyError(f"Missing column {c}")

_df = _df.rename(columns=col_map)

# Clean children category
_df['children'] = _df['children'].astype(str).str.lower()

# Basic summaries
summary = _df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])

# Proportion with any affair
_df['any_affair'] = (_df['affairs'] > 0).astype(int)
any_summary = _df.groupby('children')['any_affair'].agg(['mean', 'count'])

# Two-sample t-test (Welch)
children_yes = _df.loc[_df['children'] == 'yes', 'affairs']
children_no = _df.loc[_df['children'] == 'no', 'affairs']

t_stat, t_p = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test for robustness
u_stat, u_p = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

# Effect size (Cohen's d, unequal variances)
mean_yes = children_yes.mean()
mean_no = children_no.mean()
var_yes = children_yes.var(ddof=1)
var_no = children_no.var(ddof=1)
# Pooled SD (unbiased) with unequal sizes
n_yes = children_yes.shape[0]
n_no = children_no.shape[0]
pooled_sd = np.sqrt(((n_yes - 1) * var_yes + (n_no - 1) * var_no) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Regression models
# OLS on affairs with robust SEs
ols = smf.ols('affairs ~ C(children)', data=_df).fit(cov_type='HC3')

# Logistic regression for any affair
logit = smf.logit('any_affair ~ C(children)', data=_df).fit(disp=False)

# Poisson regression for count
poisson = smf.glm('affairs ~ C(children)', data=_df, family=sm.families.Poisson()).fit()

results = {
    'summary': summary.to_dict(),
    'any_affair': any_summary.to_dict(),
    't_test': {'t_stat': t_stat, 'p_value': t_p},
    'mann_whitney': {'u_stat': u_stat, 'p_value': u_p},
    'effect_size': {'cohen_d': cohen_d},
    'ols': {
        'coef': ols.params.to_dict(),
        'pvalues': ols.pvalues.to_dict(),
        'conf_int': ols.conf_int().to_dict(),
    },
    'logit': {
        'coef': logit.params.to_dict(),
        'pvalues': logit.pvalues.to_dict(),
        'conf_int': logit.conf_int().to_dict(),
    },
    'poisson': {
        'coef': poisson.params.to_dict(),
        'pvalues': poisson.pvalues.to_dict(),
        'conf_int': poisson.conf_int().to_dict(),
    },
}

print(json.dumps(results, indent=2))
