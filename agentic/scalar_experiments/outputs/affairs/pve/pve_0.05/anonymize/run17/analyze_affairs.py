import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Ensure expected columns

# feature2: affairs frequency; feature6: children yes/no

# Clean: drop missing
sub = df[['feature2', 'feature6']].dropna()

# Group by children

groups = {k: v['feature2'].astype(float).values for k, v in sub.groupby('feature6')}

# Ensure both groups present

if len(groups) != 2:
    raise SystemExit(f"Expected 2 groups in feature6, got {list(groups.keys())}")

# Determine order: yes/no

yes = groups.get('yes')
no = groups.get('no')

# Basic stats

def desc(arr):
    return {
        'n': int(len(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')
    }

stats_yes = desc(yes)
stats_no = desc(no)

# Welch t-test (two-sided) and one-sided (mean_yes < mean_no)

t_res = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# One-sided p-value for decrease
if np.isnan(t_res.statistic):
    p_one = float('nan')
else:
    if t_res.statistic < 0:
        p_one = t_res.pvalue / 2
    else:
        p_one = 1 - t_res.pvalue / 2

# Mann-Whitney U (two-sided)
try:
    mw = stats.mannwhitneyu(yes, no, alternative='two-sided')
    mw_p = float(mw.pvalue)
    mw_u = float(mw.statistic)
except ValueError:
    mw_p = float('nan')
    mw_u = float('nan')

# Cohen's d (yes - no)

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    s_pooled = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(s_pooled)

d = cohens_d(yes, no)

# Simple OLS with robust SE
model = smf.ols('feature2 ~ C(feature6)', data=sub).fit(cov_type='HC3')

# Extract coefficient for yes vs no (statsmodels uses baseline in alphabetical order)
# Determine param name
param_name = [p for p in model.params.index if p.startswith('C(feature6)')]
coef = model.params[param_name[0]] if param_name else float('nan')
pval = model.pvalues[param_name[0]] if param_name else float('nan')

out = {
    'stats_yes': stats_yes,
    'stats_no': stats_no,
    'mean_diff_yes_minus_no': float(stats_yes['mean'] - stats_no['mean']),
    't_stat': float(t_res.statistic),
    't_p_two_sided': float(t_res.pvalue),
    't_p_one_sided_yes_lt_no': float(p_one),
    'mannwhitney_u': mw_u,
    'mannwhitney_p_two_sided': mw_p,
    'cohens_d_yes_minus_no': float(d),
    'ols_coef_yes_minus_no': float(coef),
    'ols_pvalue': float(pval)
}

print(json.dumps(out, indent=2))
