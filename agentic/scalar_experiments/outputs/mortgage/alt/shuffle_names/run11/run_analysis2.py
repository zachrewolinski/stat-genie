import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

approval_col = 'deny'
gender_col = 'denied_PMI'

# simple logistic: approval ~ gender
sub = df[[approval_col, gender_col]].dropna()
X = sm.add_constant(sub[gender_col], has_constant='add')
y = sub[approval_col]
logit_simple = sm.Logit(y, X).fit(disp=False)

# logistic with key ratios
ratio_cols = ['mortgage_credit','housing_expense_ratio','Unnamed: 0']
sub2 = df[[approval_col, gender_col] + ratio_cols].dropna()
X2 = sm.add_constant(sub2[[gender_col]+ratio_cols], has_constant='add')
y2 = sub2[approval_col]
logit_ratio = sm.Logit(y2, X2).fit(disp=False)

print('simple coef', logit_simple.params[gender_col], 'p', logit_simple.pvalues[gender_col])
print('ratio coef', logit_ratio.params[gender_col], 'p', logit_ratio.pvalues[gender_col])
