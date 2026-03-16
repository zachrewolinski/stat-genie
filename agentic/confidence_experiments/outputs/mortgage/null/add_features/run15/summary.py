import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('mortgage.csv')

# Keep rows with non-missing female and accept

df = df[df['female'].notna() & df['accept'].notna()].copy()

# contingency table
ct = pd.crosstab(df['female'], df['accept'])
# approval rates
rates = ct.div(ct.sum(axis=1), axis=0)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)

# two-proportion z-test
p_f = rates.loc[1,1]
p_m = rates.loc[0,1]
# sizes
n_f = ct.loc[1].sum()
n_m = ct.loc[0].sum()
x_f = ct.loc[1,1]
x_m = ct.loc[0,1]

p_pool = (x_f + x_m) / (n_f + n_m)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_f + 1/n_m))
z = (p_f - p_m) / se
p_z = 2 * (1 - stats.norm.cdf(abs(z)))

# logistic regression unadjusted
model1 = smf.logit('accept ~ female', data=df).fit(disp=0)
coef1 = model1.params['female']
se1 = model1.bse['female']
p1 = model1.pvalues['female']
ci1 = model1.conf_int().loc['female']

# adjusted model
covars = ['female','black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
formula = 'accept ~ ' + ' + '.join(covars)
model2 = smf.logit(formula, data=df).fit(disp=0)
coef2 = model2.params['female']
p2 = model2.pvalues['female']
ci2 = model2.conf_int().loc['female']

# print summary values
print('n_total', len(df))
print('n_male', n_m, 'n_female', n_f)
print('accept_male', x_m, 'accept_female', x_f)
print('rate_male', p_m, 'rate_female', p_f, 'diff', p_f - p_m)
print('chi2_p', p)
print('z_p', p_z)
print('logit_unadj_coef', coef1, 'p', p1, 'OR', float(np.exp(coef1)), 'CI', list(np.exp(ci1)))
print('logit_adj_coef', coef2, 'p', p2, 'OR', float(np.exp(coef2)), 'CI', list(np.exp(ci2)))
