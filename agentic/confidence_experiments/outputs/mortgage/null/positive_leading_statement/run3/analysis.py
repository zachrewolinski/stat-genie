import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic cleanup: ensure binary as int
for col in ['female','black','self_employed','married','bad_history','deny','denied_PMI','accept']:
    if col in _df.columns:
        _df[col] = _df[col].astype(float)

# Drop rows with missing in variables used
outcome = 'accept'
base_cols = ['female']
control_cols = ['black','housing_expense_ratio','self_employed','married','mortgage_credit',
                'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']

use_cols = [outcome] + base_cols + control_cols
_df_use = _df[use_cols].dropna()

# Descriptive approval rates by gender
rates = _df_use.groupby('female')[outcome].agg(['mean','count'])
rate_female = rates.loc[1.0, 'mean'] if 1.0 in rates.index else np.nan
rate_male = rates.loc[0.0, 'mean'] if 0.0 in rates.index else np.nan

# Chi-square test of independence (female vs accept)
ct = pd.crosstab(_df_use['female'], _df_use[outcome])
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

# Logistic regression: unadjusted
X_unadj = sm.add_constant(_df_use[base_cols])
model_unadj = sm.Logit(_df_use[outcome], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Adjusted model
X_adj = sm.add_constant(_df_use[base_cols + control_cols])
model_adj = sm.Logit(_df_use[outcome], X_adj)
res_adj = model_adj.fit(disp=False)


def summarize_res(res, var):
    coef = res.params[var]
    se = res.bse[var]
    p = res.pvalues[var]
    # OR and 95% CI
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    return {
        'coef': float(coef),
        'se': float(se),
        'p': float(p),
        'or': float(or_val),
        'or_ci_low': float(ci_low),
        'or_ci_high': float(ci_high)
    }

summ_unadj = summarize_res(res_unadj, 'female')
summ_adj = summarize_res(res_adj, 'female')

# Save summary for later inspection
summary = {
    'n': int(len(_df_use)),
    'approval_rate_female': float(rate_female),
    'approval_rate_male': float(rate_male),
    'rate_diff_female_minus_male': float(rate_female - rate_male),
    'chi2_p': float(p_chi2),
    'unadjusted': summ_unadj,
    'adjusted': summ_adj,
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
