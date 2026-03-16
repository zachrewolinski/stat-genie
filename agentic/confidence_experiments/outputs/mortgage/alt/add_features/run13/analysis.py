import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'mortgage.csv'

# Load data
_df = pd.read_csv(DATA_PATH)

print('Columns:', _df.columns.tolist())
print('Shape:', _df.shape)

# basic checks
print('female value counts:', _df['female'].value_counts(dropna=False))
print('deny value counts:', _df['deny'].value_counts(dropna=False))
print('accept value counts:', _df['accept'].value_counts(dropna=False))

# Check relation accept vs deny
if 'accept' in _df.columns and 'deny' in _df.columns:
    consistent = ((1 - _df['deny']) == _df['accept']).mean()
    print('accept == 1 - deny (share):', consistent)

# Unadjusted rates
rate_table = _df.groupby('female')[['deny','accept']].mean()
count_table = _df.groupby('female')[['deny','accept']].count()
print('Unadjusted mean by female:\n', rate_table)
print('Counts by female:\n', count_table)

# Two-proportion z-test for deny difference
from statsmodels.stats.proportion import proportions_ztest

# Ensure 0/1 and drop missing
mask = _df['female'].notna() & _df['deny'].notna()
sub = _df.loc[mask, ['female','deny']]
counts = sub.groupby('female')['deny'].sum()
ns = sub.groupby('female')['deny'].count()
# female=1 vs female=0
if 0 in counts.index and 1 in counts.index:
    count = np.array([counts.loc[1], counts.loc[0]])
    nobs = np.array([ns.loc[1], ns.loc[0]])
    stat, pval = proportions_ztest(count, nobs)
    print('Two-proportion z-test deny: z=%.4f p=%.6g' % (stat, pval))
    diff = (counts.loc[1]/ns.loc[1]) - (counts.loc[0]/ns.loc[0])
    print('Deny rate female - male:', diff)

# Logistic regression (deny as outcome)
# choose mortgage-related covariates
covars = [
    'female','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio',
    'loan_to_value','denied_PMI'
]

# keep only available columns
covars = [c for c in covars if c in _df.columns]

# drop missing
reg_df = _df[covars + ['deny']].dropna()

# standardize continuous covars? not necessary for logistic; keep raw

# Build formula
formula = 'deny ~ ' + ' + '.join(covars)
print('Formula:', formula)

model = smf.logit(formula, data=reg_df).fit(disp=False)
print(model.summary())

# odds ratio for female
params = model.params
conf = model.conf_int()
if 'female' in params.index:
    coef = params['female']
    or_val = np.exp(coef)
    ci = np.exp(conf.loc['female'])
    p = model.pvalues['female']
    print('Female coef:', coef, 'OR:', or_val, 'CI:', ci.tolist(), 'p:', p)

# marginal effect at means
margeff = model.get_margeff(at='mean')
print(margeff.summary())

# Save key outputs to csv for later use
summary = {
    'n': len(reg_df),
    'female_coef': params.get('female', np.nan),
    'female_p': model.pvalues.get('female', np.nan),
    'female_or': np.exp(params.get('female', np.nan)) if 'female' in params.index else np.nan,
}
print('Summary dict:', summary)
