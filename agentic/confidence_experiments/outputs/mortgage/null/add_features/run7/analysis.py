import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('mortgage.csv')

# Determine outcome column
if 'accept' in _df.columns:
    _df['accept_outcome'] = _df['accept']
elif 'deny' in _df.columns:
    _df['accept_outcome'] = 1 - _df['deny']
else:
    raise ValueError('No accept or deny column found')

# Relevant columns for analysis
cols = [
    'accept_outcome',
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value'
]

# Keep only columns that exist
cols = [c for c in cols if c in _df.columns]

# Drop rows with missing values in relevant columns
_df_sub = _df[cols].dropna().copy()

# Basic counts
n_total = len(_df)
n_used = len(_df_sub)

# Approval rates by gender
rates = _df_sub.groupby('female')['accept_outcome'].mean()
counts = _df_sub.groupby('female')['accept_outcome'].agg(['count', 'sum'])

# Two-proportion z-test (female=1 vs male=0)
# Use normal approximation; fall back to chi-square if needed
from statsmodels.stats.proportion import proportions_ztest

if set(rates.index) >= {0, 1}:
    count = np.array([counts.loc[1, 'sum'], counts.loc[0, 'sum']])
    nobs = np.array([counts.loc[1, 'count'], counts.loc[0, 'count']])
    stat, pval = proportions_ztest(count, nobs)
else:
    stat, pval = np.nan, np.nan

# Unadjusted logistic regression
unadj_model = smf.logit('accept_outcome ~ female', data=_df_sub).fit(disp=False)

# Adjusted logistic regression
# Build formula with available covariates (excluding accept_outcome and female)
control_vars = [c for c in cols if c not in ['accept_outcome', 'female']]
formula = 'accept_outcome ~ female'
if control_vars:
    formula += ' + ' + ' + '.join(control_vars)

adj_model = smf.logit(formula, data=_df_sub).fit(disp=False)

# Extract female coefficient info
unadj_coef = unadj_model.params['female']
unadj_p = unadj_model.pvalues['female']
unadj_or = np.exp(unadj_coef)

adj_coef = adj_model.params['female']
adj_p = adj_model.pvalues['female']
adj_or = np.exp(adj_coef)

# Confidence interval for adjusted OR
adj_ci = adj_model.conf_int().loc['female']
adj_or_ci = np.exp(adj_ci)

results = {
    'n_total': int(n_total),
    'n_used': int(n_used),
    'approval_rate_female': float(rates.loc[1]) if 1 in rates.index else None,
    'approval_rate_male': float(rates.loc[0]) if 0 in rates.index else None,
    'diff_rate_female_minus_male': float((rates.loc[1] - rates.loc[0])) if set(rates.index) >= {0,1} else None,
    'ztest_stat': float(stat) if stat == stat else None,
    'ztest_p': float(pval) if pval == pval else None,
    'unadj_logit_coef_female': float(unadj_coef),
    'unadj_logit_or_female': float(unadj_or),
    'unadj_logit_p_female': float(unadj_p),
    'adj_logit_coef_female': float(adj_coef),
    'adj_logit_or_female': float(adj_or),
    'adj_logit_p_female': float(adj_p),
    'adj_logit_or_ci_lower': float(adj_or_ci[0]),
    'adj_logit_or_ci_upper': float(adj_or_ci[1]),
    'formula': formula,
    'control_vars': control_vars
}

print(results)
