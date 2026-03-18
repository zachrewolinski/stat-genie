import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
# feature2: frequency of affairs (numeric with many zeros)
# feature6: children yes/no

# basic cleaning
_df = _df.copy()

# Ensure feature6 is categorical
_df['feature6'] = _df['feature6'].astype(str).str.lower()

# Create indicator for any affair
_df['any_affair'] = _df['feature2'] > 0

# Group stats
by_children = _df.groupby('feature6')
mean_affairs = by_children['feature2'].mean()
median_affairs = by_children['feature2'].median()
prop_any_affair = by_children['any_affair'].mean()
counts = by_children.size()

# Two-sample tests for continuous (affairs frequency)
# Use nonparametric Mann-Whitney due to heavy zero inflation
children_yes = _df.loc[_df['feature6'] == 'yes', 'feature2']
children_no = _df.loc[_df['feature6'] == 'no', 'feature2']

mw_stat, mw_p = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

# t-test for means (Welch)
welch_t, welch_p = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Effect size for binary outcome: difference in proportions
prop_yes = prop_any_affair.get('yes', np.nan)
prop_no = prop_any_affair.get('no', np.nan)
prop_diff = prop_yes - prop_no

# Chi-square test for independence (any affair vs children)
contingency = pd.crosstab(_df['feature6'], _df['any_affair'])
chi2, chi2_p, chi2_dof, chi2_exp = stats.chi2_contingency(contingency)

# Logistic regression for any affair ~ children + controls
# Use other features as controls (excluding feature1 id)
# Build design matrix
# Map children yes/no to 1/0
_df['children_yes'] = (_df['feature6'] == 'yes').astype(int)

# Controls: feature3 (gender), feature4 age, feature5 yrs married, feature7 religiousness,
# feature8 education, feature9 occupation, feature10 marriage rating

# Encode gender (feature3)
_df['female'] = (_df['feature3'].astype(str).str.lower() == 'female').astype(int)

X = _df[['children_yes', 'female', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']]
X = sm.add_constant(X)

y = _df['any_affair'].astype(int)

logit_model = sm.Logit(y, X)
try:
    logit_res = logit_model.fit(disp=False)
    logit_params = logit_res.params
    logit_pvalues = logit_res.pvalues
except Exception as e:
    logit_res = None
    logit_params = None
    logit_pvalues = None


results = {
    'counts': counts.to_dict(),
    'mean_affairs': mean_affairs.to_dict(),
    'median_affairs': median_affairs.to_dict(),
    'prop_any_affair': prop_any_affair.to_dict(),
    'mannwhitney_p': float(mw_p),
    'welch_p': float(welch_p),
    'prop_diff': float(prop_diff),
    'chi2_p': float(chi2_p),
}

if logit_res is not None:
    results['logit_children_yes_coef'] = float(logit_params['children_yes'])
    results['logit_children_yes_p'] = float(logit_pvalues['children_yes'])

print(json.dumps(results, indent=2))
