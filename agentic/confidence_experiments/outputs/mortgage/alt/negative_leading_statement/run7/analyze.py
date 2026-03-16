import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic cleanup: ensure binary numeric
# female: 1 female, 0 male
# accept: 1 accepted, 0 denied

# Drop rows with missing in key fields
key_cols = ['female', 'accept']
sub = df[key_cols].dropna()

# Contingency table
ct = pd.crosstab(sub['female'], sub['accept'])
# Ensure columns 0/1 present
for col in [0,1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0,1]]

# Acceptance rates by gender
rates = ct[1] / ct.sum(axis=1)

# Chi-square test of independence
chi2, p, dof, exp = stats.chi2_contingency(ct)

# Difference in proportions (female - male) with CI
n_f = ct.loc[1].sum() if 1 in ct.index else 0
n_m = ct.loc[0].sum() if 0 in ct.index else 0
p_f = rates.loc[1] if 1 in rates.index else np.nan
p_m = rates.loc[0] if 0 in rates.index else np.nan

diff = p_f - p_m
# standard error for difference in proportions
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se

# Logistic regression: accept ~ female (unadjusted)
# Use statsmodels with robust SEs
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)

# Logistic regression: adjust for covariates to see if gender effect persists
# Select plausible covariates from dataset; use list of numeric predictors
covariates = ['black', 'housing_expense_ratio', 'self_employed', 'married',
              'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
              'loan_to_value', 'denied_PMI']

# Build formula with available columns (some may be missing in data)
avail_cov = [c for c in covariates if c in df.columns]
formula = 'accept ~ female'
if avail_cov:
    formula += ' + ' + ' + '.join(avail_cov)

model_adj = smf.logit(formula, data=df).fit(disp=False)

# Extract results
unadj_coef = model_unadj.params['female']
unadj_p = model_unadj.pvalues['female']
unadj_or = np.exp(unadj_coef)

adj_coef = model_adj.params['female']
adj_p = model_adj.pvalues['female']
adj_or = np.exp(adj_coef)

# Write out summary stats to a JSON-like dict for human reading
out = {
    'n_total': int(len(df)),
    'n_used': int(len(sub)),
    'contingency_table': ct.to_dict(),
    'acceptance_rate_female': float(p_f),
    'acceptance_rate_male': float(p_m),
    'difference_female_minus_male': float(diff),
    'diff_95ci': [float(ci_low), float(ci_high)],
    'chi2_p': float(p),
    'logit_unadj_or': float(unadj_or),
    'logit_unadj_p': float(unadj_p),
    'logit_adj_or': float(adj_or),
    'logit_adj_p': float(adj_p),
    'formula_adj': formula,
}

import json
print(json.dumps(out, indent=2))
