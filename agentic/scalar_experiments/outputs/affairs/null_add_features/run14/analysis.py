import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# Normalize children column (yes/no)
children = df['children'].astype(str).str.lower().str.strip()

# Outcome: affairs (numeric frequency scale). Also create any affair indicator.

# Basic group stats
mean_by = df.groupby(children)['affairs'].mean()
count_by = df.groupby(children)['affairs'].size()

# Any affair indicator
any_affair = (df['affairs'] > 0).astype(int)

# Proportion any affair by children
prop_by = any_affair.groupby(children).mean()

# t-test for means
yes_vals = df.loc[children == 'yes', 'affairs']
no_vals = df.loc[children == 'no', 'affairs']

# unequal variance t-test
t_stat, t_p = stats.ttest_ind(yes_vals, no_vals, equal_var=False)

# Logistic regression for any affair ~ children
# children yes as 1
children_bin = (children == 'yes').astype(int)
X = sm.add_constant(children_bin)
logit = sm.Logit(any_affair, X, missing='drop')
try:
    logit_res = logit.fit(disp=0)
    logit_coef = logit_res.params[1]
    logit_p = logit_res.pvalues[1]
except Exception:
    logit_coef = np.nan
    logit_p = np.nan

# OLS for affairs frequency
ols = sm.OLS(df['affairs'], X, missing='drop').fit()
ols_coef = ols.params[1]
ols_p = ols.pvalues[1]

# Effect size: difference in means
mean_diff = yes_vals.mean() - no_vals.mean()

# Compute scalar: negative if children reduces affairs (mean_diff < 0)
# Use effect size relative to std to scale, but cap.
pooled_std = np.sqrt((yes_vals.var(ddof=1) + no_vals.var(ddof=1)) / 2)
if pooled_std > 0:
    d = mean_diff / pooled_std
else:
    d = 0.0

# Map Cohen's d and significance to Likert-ish scale
# Base score from d scaled to 100 with |d|=1 -> 60, |d|=2 -> 90
base = np.clip(d * 60, -90, 90)
# Boost if both t-test and logit show consistent direction and p<0.05
boost = 0
if np.sign(mean_diff) < 0:
    if t_p < 0.05:
        boost -= 5
    if logit_p == logit_p and logit_p < 0.05 and logit_coef < 0:
        boost -= 5
elif np.sign(mean_diff) > 0:
    if t_p < 0.05:
        boost += 5
    if logit_p == logit_p and logit_p < 0.05 and logit_coef > 0:
        boost += 5

score = int(np.clip(round(base + boost), -100, 100))

# Output summary to stdout
print('counts:', count_by.to_dict())
print('mean_by:', mean_by.to_dict())
print('prop_by:', prop_by.to_dict())
print('mean_diff_yes_minus_no:', mean_diff)
print('t_p:', t_p)
print('logit_coef:', logit_coef, 'logit_p:', logit_p)
print('ols_coef:', ols_coef, 'ols_p:', ols_p)
print('cohen_d:', d)
print('score:', score)

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))
