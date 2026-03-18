import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure feature6 is categorical with yes/no

df['feature6'] = df['feature6'].astype(str).str.lower()

# Outcome and group

y = df['feature2']
has_children = df['feature6'] == 'yes'

# Group stats

def summary(series):
    return {
        'n': int(series.shape[0]),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std(ddof=1))
    }

stats_yes = summary(y[has_children])
stats_no = summary(y[~has_children])

# Welch t-test

t_res = stats.ttest_ind(y[has_children], y[~has_children], equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use alternative='two-sided' and handle ties with method='auto'
try:
    mw_res = stats.mannwhitneyu(y[has_children], y[~has_children], alternative='two-sided')
    mw_stat = float(mw_res.statistic)
    mw_p = float(mw_res.pvalue)
except Exception:
    mw_stat = None
    mw_p = None

# Cohen's d (pooled)

def cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if s == 0:
        return float('nan')
    return (a.mean() - b.mean()) / s

d = cohens_d(y[has_children], y[~has_children])

# OLS without controls

df['children_yes'] = has_children.astype(int)
ols_simple = smf.ols('feature2 ~ children_yes', data=df).fit(cov_type='HC3')

# OLS with controls
# Use categorical for gender (feature3)
# include age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)

ols_controls = smf.ols(
    'feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(cov_type='HC3')

# Binary outcome: any affair (>0)

df['any_affair'] = (df['feature2'] > 0).astype(int)
logit = smf.logit(
    'any_affair ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(disp=False)

# Extract

results = {
    'group_stats_yes': stats_yes,
    'group_stats_no': stats_no,
    'mean_diff_yes_minus_no': float(stats_yes['mean'] - stats_no['mean']),
    't_test': {'statistic': float(t_res.statistic), 'pvalue': float(t_res.pvalue)},
    'mw_test': {'statistic': mw_stat, 'pvalue': mw_p},
    'cohens_d': float(d),
    'ols_simple': {
        'coef_children': float(ols_simple.params['children_yes']),
        'pvalue_children': float(ols_simple.pvalues['children_yes'])
    },
    'ols_controls': {
        'coef_children': float(ols_controls.params['children_yes']),
        'pvalue_children': float(ols_controls.pvalues['children_yes'])
    },
    'logit_controls': {
        'coef_children': float(logit.params['children_yes']),
        'pvalue_children': float(logit.pvalues['children_yes'])
    }
}

print(json.dumps(results, indent=2))
