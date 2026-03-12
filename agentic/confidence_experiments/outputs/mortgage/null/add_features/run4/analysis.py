import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('mortgage.csv')

# Basic columns
print('rows', len(df))
print('columns', df.columns.tolist())

# Ensure binary outcome: deny (1 denied) and accept (1 accepted)

# Basic missingness for key columns
key_cols = ['female','deny','accept']
print(df[key_cols].isna().sum())

# basic rates
for col in ['deny','accept']:
    if col in df.columns:
        rate_female = df.loc[df['female']==1, col].mean()
        rate_male = df.loc[df['female']==0, col].mean()
        print(col, 'female', rate_female, 'male', rate_male)

# chi-square test on deny
cont = pd.crosstab(df['female'], df['deny'])
print('contingency deny')
print(cont)
chi2, p, dof, exp = stats.chi2_contingency(cont)
print('chi2 p', p)

# Logistic regression with controls
# select columns relevant to mortgage
control_cols = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI','age','occupation']
cols = ['female'] + [c for c in control_cols if c in df.columns]

# drop rows with missing in cols + outcome
model_df = df[cols + ['deny']].dropna()
print('model rows', len(model_df))

X = model_df[cols]
X = sm.add_constant(X)
y = model_df['deny']

logit = sm.Logit(y, X)
res = logit.fit(disp=False)
print(res.summary())

# odds ratio for female
coef = res.params['female']
se = res.bse['female']
# 95% CI
ci_low = coef - 1.96*se
ci_high = coef + 1.96*se
print('female coef', coef, 'se', se, 'p', res.pvalues['female'])
print('female OR', np.exp(coef), 'CI', (np.exp(ci_low), np.exp(ci_high)))

# also fit unadjusted logit
X2 = sm.add_constant(model_df[['female']])
res2 = sm.Logit(y, X2).fit(disp=False)
print('unadjusted female coef', res2.params['female'], 'p', res2.pvalues['female'], 'OR', np.exp(res2.params['female']))
