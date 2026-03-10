import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('mortgage.csv')

# Use deny as outcome (1 = denied). Approval = 1 - deny
# Exclude rows with missing female or deny

df = df.copy()

# Ensure binary

# Basic subset for gender and deny
sub = df.loc[df['female'].notna() & df['deny'].notna(), ['female','deny']].copy()
sub['female'] = sub['female'].astype(int)
sub['deny'] = sub['deny'].astype(int)

# Crosstab
ct = pd.crosstab(sub['female'], sub['deny'])

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)

# Approval rates by gender
rates = sub.groupby('female')['deny'].mean().rename('deny_rate')
approval_rates = (1 - rates)

# Two-proportion z-test for denial rates
# counts: denied by gender
count = sub.groupby('female')['deny'].sum()
# nobs: total by gender
nobs = sub.groupby('female')['deny'].count()
from statsmodels.stats.proportion import proportions_ztest
stat, p_z = proportions_ztest(count.values, nobs.values)

# Risk difference (female - male) for denial
# female=1, male=0
rate_female = rates.loc[1]
rate_male = rates.loc[0]
rd = rate_female - rate_male

# Wald CI for risk difference
# Variance of difference of proportions
var_rd = rate_female * (1 - rate_female) / nobs.loc[1] + rate_male * (1 - rate_male) / nobs.loc[0]
se_rd = np.sqrt(var_rd)
ci_low = rd - 1.96 * se_rd
ci_high = rd + 1.96 * se_rd

# Logistic regression (unadjusted)
logit_unadj = smf.logit('deny ~ female', data=sub).fit(disp=False)

# Adjusted model with key covariates
covariates = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

# Keep rows with non-missing covariates
adj_cols = ['deny', 'female'] + covariates
adj = df[adj_cols].dropna().copy()
# Ensure binary ints
adj['female'] = adj['female'].astype(int)
adj['deny'] = adj['deny'].astype(int)

# Fit adjusted logistic regression
formula = 'deny ~ female + ' + ' + '.join(covariates)
logit_adj = smf.logit(formula, data=adj).fit(disp=False)

# Extract odds ratios and CI for female

def or_ci(model, var):
    coef = model.params[var]
    se = model.bse[var]
    or_val = np.exp(coef)
    ci_l = np.exp(coef - 1.96 * se)
    ci_u = np.exp(coef + 1.96 * se)
    pval = model.pvalues[var]
    return or_val, ci_l, ci_u, pval

or_unadj = or_ci(logit_unadj, 'female')
or_adj = or_ci(logit_adj, 'female')

# Output summary
print('N_total', len(df))
print('N_sub', len(sub))
print('N_adj', len(adj))
print('Crosstab female x deny:\n', ct)
print('Denial rates by gender:', rates.to_dict())
print('Approval rates by gender:', approval_rates.to_dict())
print('Risk difference (female - male) denial:', rd)
print('RD 95% CI:', (ci_low, ci_high))
print('Chi-square p:', p)
print('Z-test p:', p_z)
print('Logit unadjusted OR (female):', or_unadj)
print('Logit adjusted OR (female):', or_adj)
