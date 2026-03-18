import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Basic cleaning / checks
# Ensure required columns exist
required = ['feature2', 'feature6', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Children indicator: yes=1, no=0
children = df['feature6'].astype(str).str.lower().map({'yes': 1, 'no': 0})
if children.isna().any():
    # If there are unexpected values, keep as NaN and drop those rows
    pass

# Outcome
outcome = pd.to_numeric(df['feature2'], errors='coerce')

# Data used for group comparisons
mask = children.notna() & outcome.notna()
children = children[mask]
outcome = outcome[mask]

# Group stats
out_yes = outcome[children == 1]
out_no = outcome[children == 0]

stats_summary = {
    'n_yes': int(out_yes.shape[0]),
    'n_no': int(out_no.shape[0]),
    'mean_yes': float(out_yes.mean()),
    'mean_no': float(out_no.mean()),
    'median_yes': float(out_yes.median()),
    'median_no': float(out_no.median()),
    'std_yes': float(out_yes.std(ddof=1)),
    'std_no': float(out_no.std(ddof=1)),
}

# Difference in means (yes - no)
mean_diff = stats_summary['mean_yes'] - stats_summary['mean_no']

# Welch's t-test
welch_t = stats.ttest_ind(out_yes, out_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use alternative='two-sided' for general difference
try:
    mwu = stats.mannwhitneyu(out_yes, out_no, alternative='two-sided')
    mwu_stat = float(mwu.statistic)
    mwu_p = float(mwu.pvalue)
except Exception:
    mwu_stat = float('nan')
    mwu_p = float('nan')

# Effect size: Cohen's d (using pooled SD)
# If either group has zero variance, set d to nan
var_yes = out_yes.var(ddof=1)
var_no = out_no.var(ddof=1)
if var_yes > 0 and var_no > 0:
    pooled_sd = np.sqrt(((out_yes.shape[0]-1)*var_yes + (out_no.shape[0]-1)*var_no) / (out_yes.shape[0] + out_no.shape[0] - 2))
    cohen_d = mean_diff / pooled_sd
else:
    cohen_d = float('nan')

# Cliff's delta (nonparametric effect size)
# Implementation: compute pairwise comparisons efficiently
# Use a fast rank-based approximation via Mann-Whitney U
if not np.isnan(mwu_stat):
    n1 = out_yes.shape[0]
    n2 = out_no.shape[0]
    cliffs_delta = (2 * mwu_stat) / (n1 * n2) - 1
else:
    cliffs_delta = float('nan')

# Regression with controls
# Build design matrix
controls = df.loc[mask, ['feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']].copy()
# Encode gender
controls['feature3'] = controls['feature3'].astype(str).str.lower().map({'male': 1, 'female': 0})

X = pd.concat([children.rename('children'), controls], axis=1)
X = sm.add_constant(X, has_constant='add')
Y = outcome

# Drop any rows with missing values in X or Y
valid = X.notna().all(axis=1) & Y.notna()
X = X.loc[valid]
Y = Y.loc[valid]

model = sm.OLS(Y, X).fit(cov_type='HC3')
coef_children = float(model.params['children'])
p_children = float(model.pvalues['children'])

results = {
    'summary': stats_summary,
    'mean_diff_yes_minus_no': float(mean_diff),
    'welch_t_stat': float(welch_t.statistic),
    'welch_p_value': float(welch_t.pvalue),
    'mwu_stat': mwu_stat,
    'mwu_p_value': mwu_p,
    'cohen_d': float(cohen_d),
    'cliffs_delta': float(cliffs_delta),
    'reg_coef_children': coef_children,
    'reg_p_children': p_children,
    'reg_n': int(model.nobs),
    'reg_r2': float(model.rsquared),
}

print(json.dumps(results, indent=2))
