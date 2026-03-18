import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map children yes/no to binary
_df['children'] = _df['feature6'].map({'yes': 1, 'no': 0})

# Outcome
_df['affairs'] = _df['feature2']

# Basic group stats
stats_by_child = _df.groupby('children')['affairs'].agg(['count','mean','median','std'])

# Welch t-test
child_yes = _df.loc[_df['children']==1, 'affairs']
child_no = _df.loc[_df['children']==0, 'affairs']

ttest = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except Exception:
    mwu = None

# Effect size (Cohen's d, Welch)
mean_yes = child_yes.mean()
mean_no = child_no.mean()
std_yes = child_yes.std(ddof=1)
std_no = child_no.std(ddof=1)
# Pooled std for Cohen's d (unbiased) using unequal sizes
n_yes = child_yes.shape[0]
n_no = child_no.shape[0]
pooled_std = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohen_d = (mean_yes - mean_no) / pooled_std

# Regression with controls
# feature3 is gender (female/male) categorical
# Use OLS with controls; treat categorical with C()
formula = 'affairs ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
model = smf.ols(formula, data=_df).fit()

# Summarize key results
results = {
    'group_stats': stats_by_child.to_dict(),
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mwu': None if mwu is None else {'statistic': float(mwu.statistic), 'pvalue': float(mwu.pvalue)},
    'effect_size': {'cohen_d': float(cohen_d)},
    'ols_children_coef': {'coef': float(model.params['children']), 'pvalue': float(model.pvalues['children'])},
    'ols_children_ci': [float(model.conf_int().loc['children',0]), float(model.conf_int().loc['children',1])],
    'n': int(_df.shape[0]),
    'n_children_yes': int(n_yes),
    'n_children_no': int(n_no),
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
