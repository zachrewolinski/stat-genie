import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'affairs.csv'

df = pd.read_csv(DATA_PATH)

# Map children yes/no to indicator
child = df['feature6'].map({'yes': 1, 'no': 0})
# Outcome: frequency of affairs
outcome = df['feature2']

# Basic group stats
stats_by_group = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# t-test (Welch)
no_vals = outcome[df['feature6'] == 'no']
yes_vals = outcome[df['feature6'] == 'yes']
t_stat, p_val = stats.ttest_ind(no_vals, yes_vals, equal_var=False, nan_policy='omit')

# Effect size: Cohen's d (using pooled SD)
# Use sample sizes and variances
n_no = no_vals.shape[0]
n_yes = yes_vals.shape[0]
var_no = np.nanvar(no_vals, ddof=1)
var_yes = np.nanvar(yes_vals, ddof=1)
pooled_sd = np.sqrt(((n_no - 1) * var_no + (n_yes - 1) * var_yes) / (n_no + n_yes - 2))
cohens_d = (np.nanmean(yes_vals) - np.nanmean(no_vals)) / pooled_sd

# OLS regression: feature2 ~ children indicator
X = sm.add_constant(child)
ols_model = sm.OLS(outcome, X, missing='drop').fit()

# Also logistic regression for any affairs (feature2 > 0)
any_affair = (outcome > 0).astype(int)
logit_model = sm.Logit(any_affair, X, missing='drop').fit(disp=0)

results = {
    'group_stats': stats_by_group.to_dict(),
    't_test': {'t_stat': float(t_stat), 'p_val': float(p_val)},
    'cohens_d_yes_minus_no': float(cohens_d),
    'ols': {
        'coef_children': float(ols_model.params[1]),
        'p_val_children': float(ols_model.pvalues[1]),
        'r2': float(ols_model.rsquared)
    },
    'logit': {
        'coef_children': float(logit_model.params[1]),
        'p_val_children': float(logit_model.pvalues[1]),
    }
}

print(json.dumps(results, indent=2))
