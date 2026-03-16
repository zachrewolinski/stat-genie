import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic variables
# Use accept (1=accepted, 0=denied)
if 'accept' not in _df.columns and 'deny' in _df.columns:
    _df['accept'] = 1 - _df['deny']

# Clean: drop rows with missing key vars
key_vars = ['female', 'accept']
# Controls for adjusted model
controls = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
]

needed = list(dict.fromkeys(key_vars + controls))

_df_clean = _df.dropna(subset=needed).copy()

# 1) Descriptive acceptance rates by gender
rates = _df_clean.groupby('female')['accept'].agg(['mean', 'count'])

# 2) Chi-square test of independence
# Contingency table: female (0/1) x accept (0/1)
cont = pd.crosstab(_df_clean['female'], _df_clean['accept'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont)

# 3) Logistic regression (unadjusted)
X_unadj = sm.add_constant(_df_clean[['female']])
model_unadj = sm.Logit(_df_clean['accept'], X_unadj).fit(disp=False)

# 4) Logistic regression (adjusted)
X_adj = sm.add_constant(_df_clean[['female'] + controls])
model_adj = sm.Logit(_df_clean['accept'], X_adj).fit(disp=False)

# Extract female effect
coef_unadj = model_unadj.params['female']
se_unadj = model_unadj.bse['female']
p_unadj = model_unadj.pvalues['female']

coef_adj = model_adj.params['female']
se_adj = model_adj.bse['female']
p_adj = model_adj.pvalues['female']

# Odds ratios and CI
from math import exp

def or_ci(coef, se, alpha=0.05):
    z = stats.norm.ppf(1 - alpha/2)
    lo = coef - z*se
    hi = coef + z*se
    return exp(coef), (exp(lo), exp(hi))

or_unadj, ci_unadj = or_ci(coef_unadj, se_unadj)
or_adj, ci_adj = or_ci(coef_adj, se_adj)

# Pack results
results = {
    'n': int(_df_clean.shape[0]),
    'accept_rate_female0': float(rates.loc[0.0, 'mean']) if 0.0 in rates.index else float(rates.loc[0, 'mean']),
    'accept_rate_female1': float(rates.loc[1.0, 'mean']) if 1.0 in rates.index else float(rates.loc[1, 'mean']),
    'count_female0': int(rates.loc[0.0, 'count']) if 0.0 in rates.index else int(rates.loc[0, 'count']),
    'count_female1': int(rates.loc[1.0, 'count']) if 1.0 in rates.index else int(rates.loc[1, 'count']),
    'chi2_p': float(p_chi2),
    'unadj_coef': float(coef_unadj),
    'unadj_p': float(p_unadj),
    'unadj_or': float(or_unadj),
    'unadj_or_ci': [float(ci_unadj[0]), float(ci_unadj[1])],
    'adj_coef': float(coef_adj),
    'adj_p': float(p_adj),
    'adj_or': float(or_adj),
    'adj_or_ci': [float(ci_adj[0]), float(ci_adj[1])],
}

print(json.dumps(results, indent=2))
