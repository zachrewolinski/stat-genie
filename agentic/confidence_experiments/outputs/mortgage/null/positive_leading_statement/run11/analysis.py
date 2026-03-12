import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')

# Basic checks
print('rows', len(df))
print('columns', df.columns.tolist())

# Ensure binary columns are numeric 0/1
# Unadjusted approval rates by gender
rate = df.groupby('female')['accept'].agg(['mean','count','sum'])
print('\nApproval rate by gender (female=1):')
print(rate)

# Difference in proportions test (z-test) for acceptance
from statsmodels.stats.proportion import proportions_ztest
count = rate['sum'].values
nobs = rate['count'].values
# order is female=0,1
stat, pval = proportions_ztest(count, nobs)
print('\nTwo-proportion z-test for acceptance rate difference (female=0 vs 1):')
print('z', stat, 'p', pval)

# Logistic regression: accept ~ female + controls
# Drop any rows with missing values
model_df = df[['accept','female','black','housing_expense_ratio','self_employed','married',
               'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']].dropna()

# Fit logit
formula = ('accept ~ female + black + housing_expense_ratio + self_employed + married '
           '+ mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI')

logit_model = smf.logit(formula, data=model_df).fit(disp=False)
print('\nLogit summary (accept outcome):')
print(logit_model.summary())

# Extract female coefficient, p-value, odds ratio, CI
coef = logit_model.params['female']
se = logit_model.bse['female']
OR = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('\nFemale coefficient:', coef)
print('Female p-value:', logit_model.pvalues['female'])
print('Female odds ratio:', OR)
print('95% CI for OR:', (ci_low, ci_high))

# Marginal effect of female on acceptance probability
marg = logit_model.get_margeff(at='overall').summary()
print('\nAverage marginal effects:')
print(marg)
