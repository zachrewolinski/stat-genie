import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

# ensure binary
# outcome: accept (1 approved)

# Basic counts
female = df['female']
accept = df['accept']

# contingency table
ct = pd.crosstab(female, accept)
# chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)

# compute approval rates by gender
rates = ct.div(ct.sum(axis=1), axis=0)

# logistic regression: accept ~ female + controls
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']

X = df[['female'] + controls].copy()
X = sm.add_constant(X)
y = accept

logit = sm.Logit(y, X, missing='drop')
res = logit.fit(disp=0)

# get female coefficient and odds ratio
coef = res.params['female']
se = res.bse['female']
pval = res.pvalues['female']
OR = np.exp(coef)
ci = res.conf_int().loc['female']
OR_ci = np.exp(ci)

# effect on probability at mean
means = X.mean()
means['female'] = 0
p0 = res.predict(means)
means['female'] = 1
p1 = res.predict(means)

# Also unadjusted difference in approval rates
rate_male = rates.loc[0,1]
rate_female = rates.loc[1,1]

# output
print('Counts by female x accept')
print(ct)
print('\nApproval rates by gender (accept=1)')
print(rates[1])
print('\nChi-square p-value:', p)
print('\nLogit female coef:', coef)
print('female OR:', OR)
print('female p-value:', pval)
print('female OR 95% CI:', OR_ci.values)
print('Predicted prob at mean male:', float(p0))
print('Predicted prob at mean female:', float(p1))
print('Unadjusted approval rate male:', float(rate_male))
print('Unadjusted approval rate female:', float(rate_female))
