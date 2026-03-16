import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic cleaning: ensure binary columns are 0/1 numeric
# Focus on female and accept; drop rows with missing values in relevant columns

# Outcome: accept (1=accepted, 0=denied)
# If accept missing, try deny and invert
if 'accept' not in _df.columns and 'deny' in _df.columns:
    _df['accept'] = 1 - _df['deny']

# define controls based on typical mortgage variables present
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
    'denied_PMI'
]

# Keep only columns that exist
controls = [c for c in controls if c in _df.columns]

cols = ['female', 'accept'] + controls

# Drop rows with missing values
_df2 = _df[cols].dropna()

# Ensure numeric
for c in cols:
    _df2[c] = pd.to_numeric(_df2[c], errors='coerce')

_df2 = _df2.dropna()

# Summary stats
n_total = len(_df2)

# Acceptance rates by gender
by_gender = _df2.groupby('female')['accept'].agg(['mean','count'])

# Chi-square test of independence on contingency table
ct = pd.crosstab(_df2['female'], _df2['accept'])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Unadjusted logistic regression
model_unadj = smf.logit('accept ~ female', data=_df2).fit(disp=False)

# Adjusted logistic regression
formula = 'accept ~ female'
if controls:
    formula += ' + ' + ' + '.join(controls)
model_adj = smf.logit(formula, data=_df2).fit(disp=False)

# Extract female effect
coef_unadj = model_unadj.params['female']
se_unadj = model_unadj.bse['female']
p_unadj = model_unadj.pvalues['female']

coef_adj = model_adj.params['female']
se_adj = model_adj.bse['female']
p_adj = model_adj.pvalues['female']

# Convert to odds ratio and 95% CI
or_unadj = float(np.exp(coef_unadj))
ci_unadj = np.exp(coef_unadj + np.array([-1,1]) * 1.96 * se_unadj)

or_adj = float(np.exp(coef_adj))
ci_adj = np.exp(coef_adj + np.array([-1,1]) * 1.96 * se_adj)

result = {
    'n_total': int(n_total),
    'accept_rate_by_gender': {
        'female_0_rate': float(by_gender.loc[0.0,'mean']) if 0.0 in by_gender.index else None,
        'female_0_n': int(by_gender.loc[0.0,'count']) if 0.0 in by_gender.index else None,
        'female_1_rate': float(by_gender.loc[1.0,'mean']) if 1.0 in by_gender.index else None,
        'female_1_n': int(by_gender.loc[1.0,'count']) if 1.0 in by_gender.index else None,
    },
    'chi2': float(chi2),
    'chi2_p': float(p_chi),
    'unadjusted': {
        'coef': float(coef_unadj),
        'p': float(p_unadj),
        'odds_ratio': float(or_unadj),
        'ci95': [float(ci_unadj[0]), float(ci_unadj[1])],
    },
    'adjusted': {
        'coef': float(coef_adj),
        'p': float(p_adj),
        'odds_ratio': float(or_adj),
        'ci95': [float(ci_adj[0]), float(ci_adj[1])],
        'controls': controls,
    }
}

print(json.dumps(result, indent=2))
