import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns by metadata names
# feature2: affair frequency; feature6: children yes/no

affairs = _df['feature2']
children = _df['feature6']

# Basic group stats
stats_by = _df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Mann-Whitney U test (non-parametric)
children_yes = _df.loc[_df['feature6']=='yes','feature2']
children_no = _df.loc[_df['feature6']=='no','feature2']

mw = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Effect size Cohen's d
mean_diff = children_yes.mean() - children_no.mean()
pooled_sd = np.sqrt(((children_yes.var(ddof=1)) + (children_no.var(ddof=1))) / 2)
cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# Binary outcome: any affair
_df['any_affair'] = (_df['feature2'] > 0).astype(int)
ct = pd.crosstab(_df['feature6'], _df['any_affair'])
chi2, p_chi2, dof, exp = stats.chi2_contingency(ct)

# Logistic regression for any affair ~ children
# Encode children yes=1 no=0
_df['children_yes'] = (_df['feature6']=='yes').astype(int)
X = sm.add_constant(_df['children_yes'])
logit_model = sm.Logit(_df['any_affair'], X).fit(disp=False)

# Output summary
output = {
    'group_stats': stats_by.to_dict(),
    'mannwhitney': {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'cohens_d': float(cohens_d),
    'any_affair_crosstab': ct.to_dict(),
    'chi2_any_affair': {'chi2': float(chi2), 'pvalue': float(p_chi2)},
    'logit_params': {k: float(v) for k, v in logit_model.params.items()},
    'logit_pvalues': {k: float(v) for k, v in logit_model.pvalues.items()},
}

print(json.dumps(output, indent=2))
