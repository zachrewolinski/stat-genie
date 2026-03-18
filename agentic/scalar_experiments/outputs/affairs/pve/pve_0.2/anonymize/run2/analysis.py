import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
# feature2: affairs frequency
# feature6: children in marriage yes/no

# Clean/prepare
_df = _df.copy()
_df['children'] = _df['feature6'].astype(str).str.lower()
_df['affairs'] = pd.to_numeric(_df['feature2'], errors='coerce')
_df = _df.dropna(subset=['children', 'affairs'])

# Split groups
with_children = _df[_df['children'] == 'yes']['affairs']
no_children = _df[_df['children'] == 'no']['affairs']

# Basic stats
stats_summary = {
    'n_yes': int(with_children.shape[0]),
    'n_no': int(no_children.shape[0]),
    'mean_yes': float(with_children.mean()),
    'mean_no': float(no_children.mean()),
    'median_yes': float(with_children.median()),
    'median_no': float(no_children.median()),
    'prop_any_yes': float((with_children > 0).mean()),
    'prop_any_no': float((no_children > 0).mean()),
}

# Welch t-test for mean difference
# H1: mean_yes < mean_no (one-sided) for decrease
# We'll compute two-sided and convert
welch_t = stats.ttest_ind(with_children, no_children, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric, one-sided)
# scipy has alternative
mw = stats.mannwhitneyu(with_children, no_children, alternative='less')

# Effect size: Cohen's d (with pooled sd using unbiased)
# Use Hedge's g? We'll compute Cohen's d with pooled SD
n1, n2 = len(with_children), len(no_children)
var1 = with_children.var(ddof=1)
var2 = no_children.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
cohens_d = (with_children.mean() - no_children.mean()) / pooled_sd if pooled_sd != 0 else np.nan

# Difference in proportions of any affair (binary)
# Use two-proportion z-test via statsmodels
from statsmodels.stats.proportion import proportions_ztest
count = np.array([(with_children > 0).sum(), (no_children > 0).sum()])
obs = np.array([n1, n2])
# alternative='smaller': p_yes < p_no
z_stat, p_prop = proportions_ztest(count, obs, alternative='smaller')

# Logistic regression for affair>0 vs children
_df['any_affair'] = (_df['affairs'] > 0).astype(int)
# Encode children yes=1, no=0
_df['children_yes'] = (_df['children'] == 'yes').astype(int)
X = sm.add_constant(_df['children_yes'])
model = sm.Logit(_df['any_affair'], X).fit(disp=False)

result = {
    'summary': stats_summary,
    'welch_t_stat': float(welch_t.statistic),
    'welch_t_pvalue_two_sided': float(welch_t.pvalue),
    'mannwhitney_u_stat': float(mw.statistic),
    'mannwhitney_pvalue_one_sided': float(mw.pvalue),
    'cohens_d_yes_minus_no': float(cohens_d),
    'prop_test_z': float(z_stat),
    'prop_test_pvalue_one_sided': float(p_prop),
    'logit_coef_children_yes': float(model.params['children_yes']),
    'logit_pvalue_children_yes': float(model.pvalues['children_yes']),
    'logit_odds_ratio_children_yes': float(np.exp(model.params['children_yes']))
}

with open('analysis_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
