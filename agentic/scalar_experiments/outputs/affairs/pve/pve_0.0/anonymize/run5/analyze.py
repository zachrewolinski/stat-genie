import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Variables
# feature2: frequency of affairs; feature6: children yes/no

affairs = _df['feature2']
children = _df['feature6']

# Basic group stats
_groups = _df.groupby('feature6')['feature2']
summary = _groups.agg(['count','mean','median','std']).to_dict()

# Proportion with any affairs (>0)
_any = _df.assign(any_affair=_df['feature2'] > 0)
prop_any = _any.groupby('feature6')['any_affair'].mean().to_dict()
counts_any = _any.groupby('feature6')['any_affair'].agg(['sum','count']).to_dict()

# Mann-Whitney U test (nonparametric)
# Use two-sided test
vals_yes = _df.loc[_df['feature6'] == 'yes', 'feature2']
vals_no = _df.loc[_df['feature6'] == 'no', 'feature2']

# Ensure non-empty
mw_res = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')

# t-test (Welch) for mean differences
welch = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d) for mean difference
mean_yes = vals_yes.mean()
mean_no = vals_no.mean()
std_yes = vals_yes.std(ddof=1)
std_no = vals_no.std(ddof=1)

n_yes = vals_yes.shape[0]
n_no = vals_no.shape[0]
# Pooled std for d (use pooled with unequal n)
pooled = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled if pooled > 0 else np.nan

# Chi-square test for any_affair vs children
contingency = pd.crosstab(_df['feature6'], _df['feature2'] > 0)
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Pack results
results = {
    'summary': summary,
    'prop_any': prop_any,
    'counts_any': {k: {kk: int(vv) for kk, vv in v.items()} for k, v in counts_any.items()},
    'mannwhitney_u': {
        'statistic': float(mw_res.statistic),
        'p_value': float(mw_res.pvalue),
    },
    'welch_ttest': {
        'statistic': float(welch.statistic),
        'p_value': float(welch.pvalue),
    },
    'cohen_d_yes_minus_no': float(cohen_d),
    'chi2_any_affair': {
        'statistic': float(chi2),
        'p_value': float(p_chi),
        'dof': int(dof)
    }
}

print(json.dumps(results, indent=2))
