import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns
# feature2: extramarital affairs frequency (numeric)
# feature6: children yes/no

# Clean/prepare
outcome = _df['feature2']
children = _df['feature6']

# Basic group stats
stats_by_group = _df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Proportion with outcome > 0
prop_gt0 = _df.assign(gt0=_df['feature2']>0).groupby('feature6')['gt0'].mean()

# Two-sample t-test (Welch)
child_yes = outcome[children=='yes']
child_no = outcome[children=='no']

ttest = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except Exception:
    mwu = None

# Logistic regression for P(affair>0) with children only
_df = _df.copy()
_df['gt0'] = _df['feature2'] > 0
_df['children_yes'] = (_df['feature6'] == 'yes').astype(int)

# Simple logistic
logit_simple = sm.Logit(_df['gt0'].astype(int), sm.add_constant(_df['children_yes']))
logit_simple_res = logit_simple.fit(disp=False)

# OLS with controls (treat outcome continuous)
# feature3 gender (category), feature4 age, feature5 years married, feature7 religiousness,
# feature8 education, feature9 occupation, feature10 marriage rating
# feature1 appears id; drop
controls = _df[['children_yes','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
# Gender as binary: male=1
controls['male'] = (_df['feature3']=='male').astype(int)
X = sm.add_constant(controls)
ols = sm.OLS(_df['feature2'], X)
ols_res = ols.fit()

# Also logistic with controls
X_logit = sm.add_constant(controls)
logit = sm.Logit(_df['gt0'].astype(int), X_logit)
logit_res = logit.fit(disp=False)

# Effect sizes
mean_diff = child_yes.mean() - child_no.mean()

def safe_ci(res, param):
    try:
        return res.conf_int().loc[param].tolist()
    except Exception:
        return [np.nan, np.nan]

results = {
    'n_yes': int(child_yes.shape[0]),
    'n_no': int(child_no.shape[0]),
    'means': stats_by_group['mean'].to_dict(),
    'medians': stats_by_group['median'].to_dict(),
    'stds': stats_by_group['std'].to_dict(),
    'prop_gt0': prop_gt0.to_dict(),
    'ttest_stat': float(ttest.statistic),
    'ttest_p': float(ttest.pvalue),
    'mwu_stat': float(mwu.statistic) if mwu is not None else None,
    'mwu_p': float(mwu.pvalue) if mwu is not None else None,
    'mean_diff_yes_minus_no': float(mean_diff),
    'logit_simple_coef_children': float(logit_simple_res.params['children_yes']),
    'logit_simple_p_children': float(logit_simple_res.pvalues['children_yes']),
    'logit_simple_ci_children': safe_ci(logit_simple_res, 'children_yes'),
    'ols_children_coef': float(ols_res.params['children_yes']),
    'ols_children_p': float(ols_res.pvalues['children_yes']),
    'ols_children_ci': safe_ci(ols_res, 'children_yes'),
    'logit_children_coef': float(logit_res.params['children_yes']),
    'logit_children_p': float(logit_res.pvalues['children_yes']),
    'logit_children_ci': safe_ci(logit_res, 'children_yes'),
}

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print(json.dumps(results, indent=2))
