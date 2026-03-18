import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns according to info.json
# feature2: frequency of affairs (numeric)
# feature6: children yes/no

# Basic cleaning
_df = _df.copy()
_df['children'] = _df['feature6'].astype(str).str.lower()
_df['affairs'] = pd.to_numeric(_df['feature2'], errors='coerce')

# Drop missing
_df = _df.dropna(subset=['children', 'affairs'])

# Group stats
_group_stats = _df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])

# Proportion with any affairs
_df['any_affair'] = (_df['affairs'] > 0).astype(int)
_prop = _df.groupby('children')['any_affair'].mean()

# Two-sample tests
# t-test (Welch)
_yes = _df.loc[_df['children'] == 'yes', 'affairs']
_no = _df.loc[_df['children'] == 'no', 'affairs']
_tstat, _tp = stats.ttest_ind(_yes, _no, equal_var=False)

# Mann-Whitney U (non-param)
_u_stat, _up = stats.mannwhitneyu(_yes, _no, alternative='two-sided')

# Difference in proportions (any affair) using chi-square
cont = pd.crosstab(_df['children'], _df['any_affair'])
_chi2, _chi2p, _, _ = stats.chi2_contingency(cont)

# Regression: OLS on affairs frequency (simple + controls)
# Create numeric covariates from other features
# feature4 age, feature5 yrs married, feature7 religiosity, feature8 education, feature9 occupation, feature10 marriage rating, feature3 gender
_df['gender'] = _df['feature3'].astype(str).str.lower()

# Encode children yes/no
_df['children_yes'] = (_df['children'] == 'yes').astype(int)

# Simple OLS
ols_simple = smf.ols('affairs ~ children_yes', data=_df).fit()

# OLS with controls (robust SE)
ols_controls = smf.ols('affairs ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(gender)', data=_df).fit(cov_type='HC3')

# Logistic regression for any affair
logit = smf.logit('any_affair ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(gender)', data=_df).fit(disp=False)

# Collect results
results = {
    'group_stats': _group_stats.to_dict(),
    'prop_any_affair': _prop.to_dict(),
    'ttest_p': float(_tp),
    'mannwhitney_p': float(_up),
    'chi2_p': float(_chi2p),
    'ols_simple_coef': float(ols_simple.params['children_yes']),
    'ols_simple_p': float(ols_simple.pvalues['children_yes']),
    'ols_controls_coef': float(ols_controls.params['children_yes']),
    'ols_controls_p': float(ols_controls.pvalues['children_yes']),
    'logit_coef': float(logit.params['children_yes']),
    'logit_p': float(logit.pvalues['children_yes']),
    'n_yes': int(_yes.shape[0]),
    'n_no': int(_no.shape[0]),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
