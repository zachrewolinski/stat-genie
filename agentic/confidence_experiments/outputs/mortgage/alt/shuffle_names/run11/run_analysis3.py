import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

approval_col = 'deny'
gender_col = 'denied_PMI'

# Rates
approval_rate = df[approval_col].mean()
rate_female = df.loc[df[gender_col]==1, approval_col].mean()
rate_male = df.loc[df[gender_col]==0, approval_col].mean()

# chi-square
cont = pd.crosstab(df[gender_col], df[approval_col])
chi2, p_chi, _, _ = stats.chi2_contingency(cont)

# simple logistic
sub = df[[approval_col, gender_col]].dropna()
X = sm.add_constant(sub[gender_col], has_constant='add')
y = sub[approval_col]
logit_simple = sm.Logit(y, X).fit(disp=False)
coef = logit_simple.params[gender_col]
se = logit_simple.bse[gender_col]

# odds ratio and 95% CI
or_simple = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# ratio-controlled logistic
ratio_cols = ['mortgage_credit','housing_expense_ratio','Unnamed: 0']
sub2 = df[[approval_col, gender_col] + ratio_cols].dropna()
X2 = sm.add_constant(sub2[[gender_col]+ratio_cols], has_constant='add')
y2 = sub2[approval_col]
logit_ratio = sm.Logit(y2, X2).fit(disp=False)
coef2 = logit_ratio.params[gender_col]
se2 = logit_ratio.bse[gender_col]

or_ratio = np.exp(coef2)
ci2_low = np.exp(coef2 - 1.96*se2)
ci2_high = np.exp(coef2 + 1.96*se2)

print('approval_rate', approval_rate)
print('rate_female', rate_female, 'rate_male', rate_male, 'diff', rate_female-rate_male)
print('chi2_p', p_chi)
print('simple OR', or_simple, 'CI', ci_low, ci_high, 'p', logit_simple.pvalues[gender_col])
print('ratio OR', or_ratio, 'CI', ci2_low, ci2_high, 'p', logit_ratio.pvalues[gender_col])
