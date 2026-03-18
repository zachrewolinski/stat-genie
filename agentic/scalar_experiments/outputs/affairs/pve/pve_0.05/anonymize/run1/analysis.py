import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns based on metadata
# feature2: frequency of extramarital intercourse
# feature6: children in marriage (yes/no)

# Basic cleaning
outcome = _df['feature2']
children = _df['feature6']

# Ensure binary grouping
children_yes = outcome[children.str.lower() == 'yes']
children_no = outcome[children.str.lower() == 'no']

# Descriptives
n_yes = children_yes.shape[0]
n_no = children_no.shape[0]
mean_yes = children_yes.mean()
mean_no = children_no.mean()
median_yes = children_yes.median()
median_no = children_no.median()

# Difference in means
mean_diff = mean_yes - mean_no

# t-test (Welch)
t_stat, t_p = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparam)
try:
    u_stat, u_p = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

# Effect size (Cohen's d)
# pooled SD for two groups
sd_yes = children_yes.std(ddof=1)
sd_no = children_no.std(ddof=1)
pooled_sd = np.sqrt(((n_yes - 1) * sd_yes**2 + (n_no - 1) * sd_no**2) / (n_yes + n_no - 2))
cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# Binary outcome: any affair (>0)
any_affair = (outcome > 0).astype(int)
children_bin = (children.str.lower() == 'yes').astype(int)

# Proportions
p_yes = any_affair[children_bin == 1].mean()
p_no = any_affair[children_bin == 0].mean()
prop_diff = p_yes - p_no

# Two-proportion z-test
count_yes = any_affair[children_bin == 1].sum()
count_no = any_affair[children_bin == 0].sum()
from statsmodels.stats.proportion import proportions_ztest
z_stat, z_p = proportions_ztest([count_yes, count_no], [n_yes, n_no])

# Logistic regression (any affair ~ children)
X = sm.add_constant(children_bin)
logit_model = sm.Logit(any_affair, X)
logit_res = logit_model.fit(disp=False)
logit_p = logit_res.pvalues['feature6']
logit_or = np.exp(logit_res.params['feature6'])

# OLS regression (frequency ~ children)
ols_model = sm.OLS(outcome, X)
ols_res = ols_model.fit()
ols_p = ols_res.pvalues['feature6']
ols_coef = ols_res.params['feature6']

results = {
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'median_yes': float(median_yes),
    'median_no': float(median_no),
    'mean_diff_yes_minus_no': float(mean_diff),
    't_p': float(t_p),
    'u_p': float(u_p),
    'cohens_d': float(cohens_d),
    'p_any_yes': float(p_yes),
    'p_any_no': float(p_no),
    'prop_diff_yes_minus_no': float(prop_diff),
    'z_p': float(z_p),
    'logit_or_yes_vs_no': float(logit_or),
    'logit_p': float(logit_p),
    'ols_coef_yes_vs_no': float(ols_coef),
    'ols_p': float(ols_p),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
