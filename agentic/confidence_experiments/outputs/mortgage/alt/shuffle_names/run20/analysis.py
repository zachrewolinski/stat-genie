import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'mortgage.csv'
df = pd.read_csv(path)
print('rows', len(df))
print('columns', df.columns.tolist())
print('binary cols counts:')
for col in df.columns:
    uniq = df[col].dropna().unique()
    if len(uniq) <= 2:
        print(col, sorted(uniq))

# Check relationship between deny and accept
if 'deny' in df.columns and 'accept' in df.columns:
    print('deny-accept crosstab')
    print(pd.crosstab(df['deny'], df['accept']))

# Basic summary for female
print('female mean', df['female'].mean())
print('deny mean', df['deny'].mean())

# Define approval as (deny == 0)
df['approved'] = (df['deny'] == 0).astype(int)

# crosstab and chi-square
ct = pd.crosstab(df['female'], df['approved'])
print('crosstab female x approved')
print(ct)
chi2, p, dof, exp = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# difference in approval rates
rates = df.groupby('female')['approved'].mean()
print('approval rates by female')
print(rates)

# Logistic regression: approved ~ female (unadjusted)
try:
    model = smf.logit('approved ~ female', data=df).fit(disp=False)
    print(model.summary())
    print('odds ratio female', np.exp(model.params['female']))
except Exception as e:
    print('logit error', e)

# Adjusted model with other covariates (excluding outcome variables)
# Use all columns except deny, approved
covariates = [c for c in df.columns if c not in ['deny','approved']]
# avoid perfect multicollinearity or categorical? We'll treat numeric as is.
formula = 'approved ~ ' + ' + '.join(covariates)
try:
    model2 = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
    print(model2.summary())
    if 'female' in model2.params:
        print('adj odds ratio female', np.exp(model2.params['female']))
        print('female p', model2.pvalues['female'])
except Exception as e:
    print('adjusted logit error', e)

