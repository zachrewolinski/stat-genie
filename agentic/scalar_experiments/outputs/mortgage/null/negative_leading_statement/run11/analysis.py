import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'

df = pd.read_csv(path)

# Basic checks
# Ensure binary columns are numeric 0/1
# Some columns might be floats; keep as is.

# Outcome: accept (1=accepted, 0=denied). If missing, derive from deny.
if 'accept' in df.columns:
    outcome = df['accept']
else:
    # if deny is 1 for denied
    outcome = 1 - df['deny']

df = df.copy()

df['accept_outcome'] = outcome

# Drop rows with missing in key columns
key_cols = ['accept_outcome', 'female']

df_key = df.dropna(subset=key_cols)

# Unadjusted comparison: acceptance rates by gender
rates = df_key.groupby('female')['accept_outcome'].mean()
counts = df_key.groupby('female')['accept_outcome'].agg(['count','sum'])

# Two-proportion z-test for difference in acceptance rates
# female=1 vs female=0
n1 = counts.loc[1, 'count'] if 1 in counts.index else 0
n0 = counts.loc[0, 'count'] if 0 in counts.index else 0
x1 = counts.loc[1, 'sum'] if 1 in counts.index else 0
x0 = counts.loc[0, 'sum'] if 0 in counts.index else 0

if n1 > 0 and n0 > 0:
    p1 = x1 / n1
    p0 = x0 / n0
    p_pool = (x1 + x0) / (n1 + n0)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
    z = (p1 - p0) / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
else:
    p1 = p0 = z = p_value = np.nan

# Adjusted logistic regression with standard covariates
# Use available covariates as controls (from description)
control_vars = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history',
    'PI_ratio', 'loan_to_value', 'denied_PMI'
]

existing_controls = [c for c in control_vars if c in df.columns]

model_df = df.dropna(subset=['accept_outcome', 'female'] + existing_controls).copy()

# Ensure numeric
for c in ['female'] + existing_controls:
    model_df[c] = pd.to_numeric(model_df[c], errors='coerce')

model_df = model_df.dropna(subset=['accept_outcome', 'female'] + existing_controls)

X = model_df[['female'] + existing_controls]
X = sm.add_constant(X, has_constant='add')

# logistic regression
logit_model = sm.Logit(model_df['accept_outcome'], X)

try:
    logit_res = logit_model.fit(disp=False)
    coef = logit_res.params['female']
    se_coef = logit_res.bse['female']
    p_coef = logit_res.pvalues['female']
    # odds ratio
    or_female = float(np.exp(coef))
    # 95% CI
    ci_low = float(np.exp(coef - 1.96 * se_coef))
    ci_high = float(np.exp(coef + 1.96 * se_coef))
except Exception as e:
    coef = se_coef = p_coef = or_female = ci_low = ci_high = np.nan

# Assemble results
results = {
    'n_total': int(df.shape[0]),
    'n_used_unadjusted': int(df_key.shape[0]),
    'accept_rate_male': float(rates.loc[0]) if 0 in rates.index else np.nan,
    'accept_rate_female': float(rates.loc[1]) if 1 in rates.index else np.nan,
    'z_test_p_value': float(p_value) if p_value is not None else np.nan,
    'n_used_adjusted': int(model_df.shape[0]),
    'logit_female_coef': float(coef) if np.isfinite(coef) else np.nan,
    'logit_female_or': float(or_female) if np.isfinite(or_female) else np.nan,
    'logit_female_ci_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
    'logit_female_ci_high': float(ci_high) if np.isfinite(ci_high) else np.nan,
    'logit_female_p_value': float(p_coef) if np.isfinite(p_coef) else np.nan,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
