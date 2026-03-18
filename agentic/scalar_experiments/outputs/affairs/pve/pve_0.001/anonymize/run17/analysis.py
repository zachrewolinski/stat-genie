import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')
# features
# feature2: affair frequency
# feature6: children yes/no

# clean
# ensure feature6 categorical
children = df['feature6'].astype(str)
# affair
affairs = pd.to_numeric(df['feature2'], errors='coerce')

# drop missing
mask = affairs.notna() & children.notna()
children = children[mask]
affairs = affairs[mask]

# group stats
stats_by = affairs.groupby(children).agg(['count','mean','median','std'])

# difference in means (yes - no)
mean_yes = stats_by.loc['yes','mean']
mean_no = stats_by.loc['no','mean']
mean_diff = mean_yes - mean_no

# t-test (Welch)
yes_vals = affairs[children == 'yes']
no_vals = affairs[children == 'no']

# If either group small, t-test might be unreliable but ok.
welch = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use alternative='two-sided' in scipy
mw = stats.mannwhitneyu(yes_vals, no_vals, alternative='two-sided')

# OLS regression (affairs ~ children indicator)
# children indicator: yes=1, no=0
child_ind = (children == 'yes').astype(int)
X = sm.add_constant(child_ind)
ols = sm.OLS(affairs, X).fit(cov_type='HC3')

# logistic for any affair >0
any_affair = (affairs > 0).astype(int)
logit = sm.Logit(any_affair, X).fit(disp=False)

# effect sizes
# Cohen's d for difference in means (pooled SD)
# use Welch's d with pooled standardizer
n1 = yes_vals.shape[0]
n0 = no_vals.shape[0]
var1 = yes_vals.var(ddof=1)
var0 = no_vals.var(ddof=1)
# pooled SD (unbiased)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0)/(n1+n0-2))
cohen_d = (mean_yes - mean_no) / pooled_sd

output = {
    "group_stats": stats_by.to_dict(),
    "mean_diff_yes_minus_no": mean_diff,
    "welch_t": {"stat": float(welch.statistic), "p": float(welch.pvalue)},
    "mannwhitneyu": {"stat": float(mw.statistic), "p": float(mw.pvalue)},
    "ols": {"coef_child": float(ols.params[1]), "p": float(ols.pvalues[1]), "r2": float(ols.rsquared)},
    "logit": {"coef_child": float(logit.params[1]), "p": float(logit.pvalues[1])},
    "cohen_d": float(cohen_d),
}

print(json.dumps(output, indent=2))
