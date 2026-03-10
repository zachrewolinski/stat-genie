import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv('mortgage.csv')

# Ensure key columns exist
required_cols = ['female', 'accept']
for col in required_cols:
    if col not in _df.columns:
        raise ValueError(f"Missing column: {col}")

# Use accept as outcome; drop rows with missing in key columns
_df = _df.copy()

# Coerce to numeric just in case
_df['female'] = pd.to_numeric(_df['female'], errors='coerce')
_df['accept'] = pd.to_numeric(_df['accept'], errors='coerce')

base = _df.dropna(subset=['female', 'accept']).copy()

# Keep only 0/1 in female and accept
base = base[(base['female'].isin([0, 1])) & (base['accept'].isin([0, 1]))]

# Simple association: acceptance rate by gender
ct = pd.crosstab(base['female'], base['accept'])
# Ensure columns for 0 and 1 exist
for col in [0, 1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0, 1]]

# counts
n_female = ct.loc[1].sum() if 1 in ct.index else 0
n_male = ct.loc[0].sum() if 0 in ct.index else 0

accept_female = ct.loc[1, 1] if 1 in ct.index else 0
accept_male = ct.loc[0, 1] if 0 in ct.index else 0

rate_female = accept_female / n_female if n_female else np.nan
rate_male = accept_male / n_male if n_male else np.nan
rate_diff = rate_female - rate_male

# Chi-square test of independence
chi2, chi_p, dof, expected = stats.chi2_contingency(ct.values)

# Two-proportion z-test
count = np.array([accept_female, accept_male])
ns = np.array([n_female, n_male])
if np.all(ns > 0):
    zstat, z_p = proportions_ztest(count, ns)
else:
    zstat, z_p = np.nan, np.nan

# Odds ratio (female vs male)
# add 0.5 continuity correction if any cell zero
a = accept_female
b = n_female - accept_female
c = accept_male
d = n_male - accept_male
if min(a, b, c, d) == 0:
    a += 0.5; b += 0.5; c += 0.5; d += 0.5
odds_ratio = (a / b) / (c / d)

# Logistic regression (unadjusted)
logit_unadj = smf.logit('accept ~ female', data=base).fit(disp=False)

# Adjusted model with core underwriting controls if available
control_cols = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

available_controls = [c for c in control_cols if c in base.columns]

adj_data = base.dropna(subset=available_controls + ['female', 'accept'])
# Ensure numeric
for c in available_controls:
    adj_data[c] = pd.to_numeric(adj_data[c], errors='coerce')
adj_data = adj_data.dropna(subset=available_controls)

if available_controls:
    formula = 'accept ~ female + ' + ' + '.join(available_controls)
    logit_adj = smf.logit(formula, data=adj_data).fit(disp=False)
else:
    logit_adj = None

results = {
    'n_total': int(base.shape[0]),
    'n_female': int(n_female),
    'n_male': int(n_male),
    'accept_female': int(accept_female),
    'accept_male': int(accept_male),
    'rate_female': float(rate_female),
    'rate_male': float(rate_male),
    'rate_diff_female_minus_male': float(rate_diff),
    'chi2_p': float(chi_p),
    'ztest_p': float(z_p),
    'odds_ratio_female_vs_male': float(odds_ratio),
    'logit_unadj_coef_female': float(logit_unadj.params['female']),
    'logit_unadj_p_female': float(logit_unadj.pvalues['female']),
}

if logit_adj is not None:
    results.update({
        'logit_adj_n': int(adj_data.shape[0]),
        'logit_adj_coef_female': float(logit_adj.params['female']),
        'logit_adj_p_female': float(logit_adj.pvalues['female']),
    })

print(json.dumps(results, indent=2))
