import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic cleaning
# Ensure binary columns are numeric
for col in ['female','black','self_employed','married','bad_history','deny','accept','denied_PMI']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Define outcome: accept (1 accepted, 0 denied)
# If accept missing, derive from deny
if 'accept' in _df.columns and _df['accept'].notna().any():
    _df['accept_bin'] = _df['accept']
elif 'deny' in _df.columns:
    _df['accept_bin'] = 1 - _df['deny']
else:
    raise ValueError('No accept/deny column found')

# Drop rows with missing in key vars
key_vars = ['accept_bin','female']
_df1 = _df.dropna(subset=key_vars)

# Unadjusted approval rates by gender
rates = _df1.groupby('female')['accept_bin'].agg(['mean','count'])

# Chi-square test
cont = pd.crosstab(_df1['female'], _df1['accept_bin'])
chi2, p_chi, dof, expected = stats.chi2_contingency(cont)

# Logistic regression - unadjusted
logit_unadj = smf.logit('accept_bin ~ female', data=_df1).fit(disp=False)

# Adjusted model with creditworthiness and risk controls
candidate_covs = [
    'black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value'
]
# Keep only covariates that exist and have variation
covs = []
for c in candidate_covs:
    if c in _df.columns:
        s = pd.to_numeric(_df[c], errors='coerce')
        if s.nunique(dropna=True) > 1:
            covs.append(c)

model_df = _df1.dropna(subset=covs)

formula = 'accept_bin ~ female'
if covs:
    formula += ' + ' + ' + '.join(covs)

logit_adj = smf.logit(formula, data=model_df).fit(disp=False)

# Extract key stats
results = {
    'n_total': int(_df1.shape[0]),
    'n_adj': int(model_df.shape[0]),
    'approval_rate_male': float(rates.loc[0.0,'mean']) if 0.0 in rates.index else float(rates.loc[0,'mean']) if 0 in rates.index else None,
    'approval_rate_female': float(rates.loc[1.0,'mean']) if 1.0 in rates.index else float(rates.loc[1,'mean']) if 1 in rates.index else None,
    'count_male': int(rates.loc[0.0,'count']) if 0.0 in rates.index else int(rates.loc[0,'count']) if 0 in rates.index else None,
    'count_female': int(rates.loc[1.0,'count']) if 1.0 in rates.index else int(rates.loc[1,'count']) if 1 in rates.index else None,
    'chi2_p': float(p_chi),
    'unadj_coef_female': float(logit_unadj.params['female']),
    'unadj_se_female': float(logit_unadj.bse['female']),
    'unadj_p_female': float(logit_unadj.pvalues['female']),
    'unadj_or_female': float(np.exp(logit_unadj.params['female'])),
    'adj_coef_female': float(logit_adj.params['female']),
    'adj_se_female': float(logit_adj.bse['female']),
    'adj_p_female': float(logit_adj.pvalues['female']),
    'adj_or_female': float(np.exp(logit_adj.params['female'])),
    'covariates_used': covs,
    'formula_adj': formula
}

# Print results
import json
print(json.dumps(results, indent=2))
