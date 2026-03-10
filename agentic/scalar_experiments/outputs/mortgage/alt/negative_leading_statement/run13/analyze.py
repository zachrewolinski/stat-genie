import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Basic checks
print('rows', len(df))
print(df[['female','accept','deny']].head())

# Ensure accept is 1 if accepted, 0 if denied
# check consistency
consistency = ((df['accept'] + df['deny']) == 1).mean()
print('accept+deny==1 proportion', consistency)

# Unadjusted approval rates by gender
rates = df.groupby('female')['accept'].mean()
print('approval rates by female', rates)

# Difference in proportions test
# contingency table
ct = pd.crosstab(df['female'], df['accept'])
print('crosstab\n', ct)
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Proportion difference and CI (Wald)
# female=1 vs female=0
n1 = ct.loc[1].sum()
n0 = ct.loc[0].sum()
# accept counts
x1 = ct.loc[1, 1]
x0 = ct.loc[0, 1]

p1 = x1 / n1
p0 = x0 / n0
diff = p1 - p0
# standard error
se = (p1*(1-p1)/n1 + p0*(1-p0)/n0) ** 0.5
z = diff / se if se>0 else float('nan')
# 95% CI
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se
print('diff (female - male) approval', diff, 'SE', se, 'z', z, 'CI', (ci_low, ci_high))

# Logistic regression unadjusted
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)
print(model_unadj.summary())

# Logistic regression adjusted with covariates (excluding target leak like deny)
# Use variables likely available to lenders: black, housing_expense_ratio, self_employed, married,
# mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI
# note: accept and deny are complements, so exclude deny.
formula = 'accept ~ female + black + housing_expense_ratio + self_employed + married + mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI'
model_adj = smf.logit(formula, data=df).fit(disp=False)
print(model_adj.summary())

# Odds ratio for female
params = model_adj.params
conf = model_adj.conf_int()

or_female = float(params['female'].round(6))
# convert to odds ratio
import numpy as np
or_val = float(np.exp(params['female']))
or_ci = np.exp(conf.loc['female'].values)
print('female log-odds', params['female'])
print('female OR', or_val, 'CI', or_ci, 'p', model_adj.pvalues['female'])

# Also compute marginal effect at means
mfx = model_adj.get_margeff(at='mean').summary()
print(mfx)

