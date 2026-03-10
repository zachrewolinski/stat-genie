import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
df = pd.read_csv('mortgage.csv')

# Based on info.json descriptions:
# gender variable is column 'denied_PMI' (1 if female, 0 if male)
# approval variable is column 'deny' (1 if accepted, 0 if denied)

gender_col = 'denied_PMI'
approval_col = 'deny'

# Basic counts
print('gender counts', df[gender_col].value_counts().to_dict())
print('approval counts', df[approval_col].value_counts().to_dict())

# Approval rate by gender
rate_by_gender = df.groupby(gender_col)[approval_col].mean()
print('approval rate by gender')
print(rate_by_gender)

# Two-proportion z-test for approval rate difference
ct = df.groupby(gender_col)[approval_col].agg(['sum','count'])
if 0 in ct.index and 1 in ct.index:
    count = ct['sum'].loc[[0,1]].to_numpy()
    nobs = ct['count'].loc[[0,1]].to_numpy()
    stat, pval = proportions_ztest(count, nobs)
    print('ztest p', pval, 'stat', stat)

# Unadjusted logistic regression: approval ~ gender
X = sm.add_constant(df[[gender_col]])
y = df[approval_col]
model_unadj = sm.Logit(y, X).fit(disp=False)
print('unadj coef', model_unadj.params[gender_col], 'p', model_unadj.pvalues[gender_col])

# Adjusted logistic regression with other covariates (all numeric columns except approval)
# Use all remaining columns as controls
controls = [c for c in df.columns if c != approval_col]
X_all = df[controls].apply(pd.to_numeric, errors='coerce')
X_all = sm.add_constant(X_all, has_constant='add')
# drop missing rows
data = pd.concat([df[approval_col], X_all], axis=1).dropna()
y2 = data[approval_col]
X2 = data.drop(columns=[approval_col])

# Fit logistic regression; try regularization if needed
try:
    model_adj = sm.Logit(y2, X2).fit(disp=False)
    coef = model_adj.params[gender_col]
    pval = model_adj.pvalues[gender_col]
    print('adj coef', coef, 'p', pval)
except Exception as e:
    print('adj logit failed', e)
    # fallback to penalized fit
    try:
        model_adj = sm.Logit(y2, X2).fit_regularized(disp=False)
        coef = model_adj.params[gender_col]
        print('adj coef (regularized)', coef)
    except Exception as e2:
        print('regularized failed', e2)

