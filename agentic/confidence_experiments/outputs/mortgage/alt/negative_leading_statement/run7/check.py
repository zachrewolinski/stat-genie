import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')

covariates = ['black', 'housing_expense_ratio', 'self_employed', 'married',
              'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
              'loan_to_value', 'denied_PMI']

avail_cov = [c for c in covariates if c in df.columns]
formula = 'accept ~ female'
if avail_cov:
    formula += ' + ' + ' + '.join(avail_cov)

model_adj = smf.logit(formula, data=df).fit(disp=False)

# statsmodels uses complete cases
used_idx = model_adj.model.data.row_labels
used = df.loc[used_idx]

ct = pd.crosstab(used['female'], used['accept'])
for col in [0,1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0,1]]

rates = ct[1] / ct.sum(axis=1)
chi2, p, dof, exp = stats.chi2_contingency(ct)

n_f = ct.loc[1].sum() if 1 in ct.index else 0
n_m = ct.loc[0].sum() if 0 in ct.index else 0
p_f = rates.loc[1] if 1 in rates.index else np.nan
p_m = rates.loc[0] if 0 in rates.index else np.nan

diff = p_f - p_m
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se

print('nobs', int(model_adj.nobs))
print('accept_rate_female', p_f)
print('accept_rate_male', p_m)
print('diff', diff)
print('ci', (ci_low, ci_high))
print('chi2_p', p)
