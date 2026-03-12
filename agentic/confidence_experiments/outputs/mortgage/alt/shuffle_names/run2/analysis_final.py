import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.api as sm

# Load data
df = pd.read_csv('mortgage.csv')

# Variable mapping from info.json descriptions
# gender: column 'denied_PMI' (1 female, 0 male)
# approval: column 'deny' (1 accepted, 0 denied)

gender_col = 'denied_PMI'
approval_col = 'deny'

# Clean data
sub = df[[gender_col, approval_col]].dropna()

# Counts
counts = sub.groupby(gender_col)[approval_col].agg(['sum','count'])
# ensure ordering: male=0, female=1
count_m = counts.loc[0, 'sum']
count_f = counts.loc[1, 'sum']
n_m = counts.loc[0, 'count']
n_f = counts.loc[1, 'count']

p_m = count_m / n_m
p_f = count_f / n_f

diff = p_f - p_m

# z-test for difference in proportions
stat, pval = proportions_ztest([count_f, count_m], [n_f, n_m])

# 95% CI for difference in proportions
ci_low, ci_high = confint_proportions_2indep(count_f, n_f, count_m, n_m, method='wald')

# Unadjusted logistic regression (approval ~ female)
X = sm.add_constant(sub[[gender_col]], has_constant='add')
y = sub[approval_col]
model = sm.Logit(y, X).fit(disp=False)
coef = model.params[gender_col]
coef_p = model.pvalues[gender_col]

# Output summary
print('n_male', n_m, 'n_female', n_f)
print('approval_male', p_m)
print('approval_female', p_f)
print('diff_female_minus_male', diff)
print('z_p', pval)
print('diff_ci', ci_low, ci_high)
print('logit_coef', coef, 'logit_p', coef_p)

