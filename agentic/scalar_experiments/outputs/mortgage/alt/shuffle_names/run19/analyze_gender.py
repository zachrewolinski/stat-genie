import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Based on info.json descriptions, the gender indicator is in column 'denied_PMI'
# and the denial indicator is in column 'self_employed' (1=denied, 0=accepted).
# The column 'deny' is the complement of 'self_employed' (1=accepted).

gender_col = 'denied_PMI'
deny_col = 'self_employed'
accept_col = 'deny'

# Sanity check: accept + deny should be 1
if not np.allclose(df[accept_col] + df[deny_col], 1):
    raise ValueError('Expected accept and deny columns to be complements.')

# Use acceptance as outcome for the research question
outcome_col = accept_col

# Contingency table and rates
ct = pd.crosstab(df[gender_col], df[outcome_col])
for val in [0, 1]:
    if val not in ct.columns:
        ct[val] = 0
ct = ct[[0, 1]]

rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else np.nan
rate_diff = rate_female - rate_male

# Chi-square test
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

base = df[[outcome_col, gender_col]].dropna()
# Unadjusted logistic regression
X = sm.add_constant(base[gender_col])
model = sm.Logit(base[outcome_col], X).fit(disp=False)
coef = model.params[gender_col]
p_logit = model.pvalues[gender_col]
or_val = float(np.exp(coef))

# Adjusted logistic regression with numeric controls
candidates = [c for c in df.columns if c not in [gender_col, outcome_col, deny_col]]
# keep numeric columns
numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
# Remove any binary columns that are perfectly collinear with outcome (already removed deny_col)
# Use all numeric controls with low missingness
valid_cols = [c for c in numeric_cols if df[c].isna().mean() < 0.05]

X_full = df[[gender_col] + valid_cols].copy()
full = pd.concat([df[outcome_col], X_full], axis=1).dropna()

y = full[outcome_col]
X_full = sm.add_constant(full.drop(columns=[outcome_col]))

adj_result = {}
try:
    model_full = sm.Logit(y, X_full).fit(disp=False)
    coef_full = model_full.params.get(gender_col, np.nan)
    p_full = model_full.pvalues.get(gender_col, np.nan)
    or_full = float(np.exp(coef_full)) if pd.notna(coef_full) else np.nan
    adj_result = {
        'n': int(model_full.nobs),
        'female_coef': float(coef_full),
        'female_p': float(p_full),
        'female_or': float(or_full),
        'num_controls': len(valid_cols),
        'controls': valid_cols,
    }
except Exception as e:
    adj_result = {'error': str(e), 'num_controls': len(valid_cols), 'controls': valid_cols}

output = {
    'gender_col': gender_col,
    'accept_col': outcome_col,
    'deny_col': deny_col,
    'contingency': ct.to_dict(),
    'accept_rate_female': float(rate_female),
    'accept_rate_male': float(rate_male),
    'accept_rate_diff': float(rate_diff),
    'chi2_p': float(p_chi2),
    'logit_coef': float(coef),
    'logit_p': float(p_logit),
    'odds_ratio': float(or_val),
    'adjusted': adj_result,
}

print(json.dumps(output, indent=2))
