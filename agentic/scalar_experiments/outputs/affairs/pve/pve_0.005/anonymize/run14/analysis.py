import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
children = _df['feature6'].astype(str).str.lower()
child_yes = (children == 'yes').astype(int)
child_yes.name = 'child_yes'
affairs = pd.to_numeric(_df['feature2'], errors='coerce')

# Basic group stats
summary = _df.assign(child_yes=child_yes, affairs=affairs).groupby('child_yes')['affairs'].agg(
    ['count', 'mean', 'median', 'std']
)

# Welch t-test
no_affairs = affairs[child_yes == 0]
yes_affairs = affairs[child_yes == 1]
welch = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use ranks; if there are many ties, still ok.
mann = stats.mannwhitneyu(yes_affairs, no_affairs, alternative='two-sided')

# Effect size (Cohen's d) for difference in means
mean_diff = yes_affairs.mean() - no_affairs.mean()
# pooled SD using unequal sizes (Hedges' g not necessary here)
ny = yes_affairs.count()
nn = no_affairs.count()
# avoid division by zero
if ny > 1 and nn > 1:
    pooled_sd = np.sqrt(((ny - 1) * yes_affairs.var(ddof=1) + (nn - 1) * no_affairs.var(ddof=1)) / (ny + nn - 2))
    cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan
else:
    cohen_d = np.nan

# OLS with robust SE: affairs ~ child_yes
X = sm.add_constant(child_yes)
ols = sm.OLS(affairs, X, missing='drop').fit(cov_type='HC3')

# Logistic regression for any affair (>0)
any_affair = (affairs > 0).astype(int)
logit = sm.Logit(any_affair, X, missing='drop').fit(disp=False)

# Odds ratio for child_yes (using default MLE standard errors)
coef = logit.params[1]
se = logit.bse[1]
# 95% CI for OR
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
or_val = np.exp(coef)
or_low = np.exp(ci_low)
or_high = np.exp(ci_high)

out = {
    'summary': summary.reset_index().to_dict(orient='list'),
    'welch_t': {'stat': float(welch.statistic), 'pvalue': float(welch.pvalue)},
    'mannwhitneyu': {'stat': float(mann.statistic), 'pvalue': float(mann.pvalue)},
    'mean_diff_yes_minus_no': float(mean_diff),
    'cohen_d': float(cohen_d),
    'ols_coef_child_yes': float(ols.params['child_yes']),
    'ols_pvalue_child_yes': float(ols.pvalues['child_yes']),
    'logit_coef_child_yes': float(coef),
    'logit_pvalue_child_yes': float(logit.pvalues[1]),
    'logit_or_child_yes': float(or_val),
    'logit_or_ci95': [float(or_low), float(or_high)],
    'n_rows': int(_df.shape[0]),
}

print(json.dumps(out, indent=2))
