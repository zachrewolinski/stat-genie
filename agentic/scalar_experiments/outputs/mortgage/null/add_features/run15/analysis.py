import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('mortgage.csv')

# Basic checks
print(df.head())
print(df.columns)
print(df['female'].value_counts(dropna=False))
print(df['accept'].value_counts(dropna=False))

# Contingency table for female vs accept
ct = pd.crosstab(df['female'], df['accept'])
print('contingency:\n', ct)

# approval rate by gender
rates = ct.div(ct.sum(axis=1), axis=0)
print('rates:\n', rates)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# difference in proportions and z-test
# compute proportion accept for female and male
# female=1, male=0
p_f = rates.loc[1,1]
p_m = rates.loc[0,1]
print('p_f', p_f, 'p_m', p_m, 'diff', p_f - p_m)

# two-proportion z-test
n_f = ct.loc[1].sum()
n_m = ct.loc[0].sum()
# successes
x_f = ct.loc[1,1]
x_m = ct.loc[0,1]
# pooled proportion
p_pool = (x_f + x_m) / (n_f + n_m)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_f + 1/n_m))
z = (p_f - p_m) / se
p_z = 2 * (1 - stats.norm.cdf(abs(z)))
print('z', z, 'p_z', p_z)

# Logistic regression unadjusted
# using accept ~ female
model1 = smf.logit('accept ~ female', data=df).fit(disp=0)
print(model1.summary())

# Adjusted model with common covariates (credit and financial variables)
# Choose variables likely relevant: black, housing_expense_ratio, self_employed, married,
# mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI
# maybe age? Not sure. We'll include those from mortgage dataset.

covars = ['female','black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
formula = 'accept ~ ' + ' + '.join(covars)
model2 = smf.logit(formula, data=df).fit(disp=0)
print(model2.summary())

# compute odds ratio for female in adjusted model
params = model2.params
conf = model2.conf_int()
print('female coef', params['female'])
print('female OR', np.exp(params['female']))
print('female OR 95% CI', np.exp(conf.loc['female']))

# robust check with deny as outcome? accept is 1 accept. gender effect on denial too.
model3 = smf.logit('deny ~ female', data=df).fit(disp=0)
print(model3.summary())
