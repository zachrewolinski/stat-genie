import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('affairs.csv')

# children indicator: feature6 yes/no
# affair engagement: feature2 (continuous)

# basic group stats
stats_by = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# proportion of any affair (>0)
any_affair = (df['feature2'] > 0).astype(int)
any_affair.name = 'any_affair'
prop_by = df.assign(any_affair=any_affair).groupby('feature6')['any_affair'].mean()

# Welch t-test for difference in means
vals_yes = df.loc[df['feature6']=='yes','feature2']
vals_no = df.loc[df['feature6']=='no','feature2']

ttest = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    mw = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
except Exception:
    mw = None

# Cohen's d for difference in means (yes - no)
# pooled SD with unequal sample sizes
n1, n2 = len(vals_yes), len(vals_no)
var1, var2 = vals_yes.var(ddof=1), vals_no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
cohens_d = (vals_yes.mean() - vals_no.mean()) / pooled_sd if pooled_sd > 0 else np.nan

# Logistic regression for any affair ~ children
X = pd.get_dummies(df['feature6'], drop_first=True)
# If categories are yes/no, drop_first yields column for 'yes' (assuming 'no' is baseline)
X = sm.add_constant(X)
logit_model = sm.Logit(any_affair, X)
try:
    logit_res = logit_model.fit(disp=False)
    logit_p = float(logit_res.pvalues[1]) if len(logit_res.pvalues) > 1 else np.nan
    logit_coef = float(logit_res.params[1]) if len(logit_res.params) > 1 else np.nan
    logit_or = float(np.exp(logit_coef)) if np.isfinite(logit_coef) else np.nan
except Exception:
    logit_res = None
    logit_p = np.nan
    logit_coef = np.nan
    logit_or = np.nan

output = {
    'stats_by': stats_by.to_dict(),
    'prop_any_affair_by': prop_by.to_dict(),
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitney': None if mw is None else {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'cohens_d_yes_minus_no': float(cohens_d),
    'logit_any_affair_children_yes': {'coef': logit_coef, 'odds_ratio': logit_or, 'pvalue': logit_p},
}

with open('analysis_results.json','w') as f:
    json.dump(output,f,indent=2)

print(json.dumps(output, indent=2))
