import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Map children yes/no
children = df['feature6'].astype(str)

affairs = df['feature2']

# groups
mask_yes = children.str.lower() == 'yes'
mask_no = children.str.lower() == 'no'

y = affairs[mask_yes]
n = affairs[mask_no]

# Basic stats
summary = {
    'n_yes': int(y.shape[0]),
    'n_no': int(n.shape[0]),
    'mean_yes': float(np.mean(y)),
    'mean_no': float(np.mean(n)),
    'median_yes': float(np.median(y)),
    'median_no': float(np.median(n)),
    'std_yes': float(np.std(y, ddof=1)),
    'std_no': float(np.std(n, ddof=1)),
}

# Welch's t-test
welch = stats.ttest_ind(y, n, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mw = stats.mannwhitneyu(y, n, alternative='two-sided')

# Effect size (Cohen's d) using pooled sd
n1, n2 = len(y), len(n)
var1, var2 = np.var(y, ddof=1), np.var(n, ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2))
cohen_d = (np.mean(y)-np.mean(n))/pooled_sd if pooled_sd > 0 else np.nan

# OLS regression with robust SE
X = sm.add_constant(mask_yes.astype(int))
model = sm.OLS(affairs, X).fit(cov_type='HC3')

print('SUMMARY', summary)
print('WELCH', {'stat': float(welch.statistic), 'p': float(welch.pvalue)})
print('MANN_WHITNEY', {'stat': float(mw.statistic), 'p': float(mw.pvalue)})
print('COHEN_D', float(cohen_d))
print('OLS_COEF', {'coef_children_yes': float(model.params[1]), 'p': float(model.pvalues[1])})

# Compute 95% CI for mean difference
mean_diff = np.mean(y) - np.mean(n)
se_diff = np.sqrt(var1/n1 + var2/n2)
ci_low = mean_diff - stats.t.ppf(0.975, df=min(n1-1, n2-1)) * se_diff
ci_high = mean_diff + stats.t.ppf(0.975, df=min(n1-1, n2-1)) * se_diff
print('MEAN_DIFF', {'diff': float(mean_diff), 'ci_low': float(ci_low), 'ci_high': float(ci_high)})
