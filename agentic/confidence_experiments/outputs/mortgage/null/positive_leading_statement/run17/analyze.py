import json
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic prep
# Outcome: accept (1=accepted, 0=denied)
# Predictor: female (1=female, 0=male)

# Ensure numeric
for col in _df.columns:
    if _df[col].dtype == object:
        _df[col] = pd.to_numeric(_df[col], errors='ignore')

# Drop rows with missing values in relevant columns

# Unadjusted approval rates

group = _df.groupby('female')['accept']
counts = group.count()
means = group.mean()

# Two-proportion z-test for acceptance rates
# female=1 vs female=0

n_f = counts.get(1.0, 0)
n_m = counts.get(0.0, 0)
acc_f = means.get(1.0, 0)
acc_m = means.get(0.0, 0)

# If there are any missing or zero counts, guard
if n_f > 0 and n_m > 0:
    # compute z-test
    p1 = acc_f
    p2 = acc_m
    p_pool = (p1 * n_f + p2 * n_m) / (n_f + n_m)
    se = (p_pool * (1 - p_pool) * (1/n_f + 1/n_m)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = None
    p_value = None

# Unadjusted logistic regression

subset_cols = ['accept', 'female']
_df_unadj = _df[subset_cols].dropna()

X_unadj = sm.add_constant(_df_unadj['female'])
model_unadj = sm.Logit(_df_unadj['accept'], X_unadj)
result_unadj = model_unadj.fit(disp=0)

# Adjusted logistic regression: include relevant controls
control_cols = [
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

adj_cols = ['accept', 'female'] + control_cols
_df_adj = _df[adj_cols].dropna()
X_adj = sm.add_constant(_df_adj[['female'] + control_cols])
model_adj = sm.Logit(_df_adj['accept'], X_adj)
result_adj = model_adj.fit(disp=0)

# Extract stats

unadj_coef = result_unadj.params['female']
unadj_p = result_unadj.pvalues['female']

adj_coef = result_adj.params['female']
adj_p = result_adj.pvalues['female']

# Convert to odds ratios
unadj_or = float(pd.Series([unadj_coef]).apply(lambda x: pd.np.exp(x)).iloc[0])
adj_or = float(pd.Series([adj_coef]).apply(lambda x: pd.np.exp(x)).iloc[0])

# Output summary
summary = {
    'n_total': int(len(_df)),
    'n_female': int(n_f),
    'n_male': int(n_m),
    'accept_rate_female': float(acc_f),
    'accept_rate_male': float(acc_m),
    'accept_rate_diff_female_minus_male': float(acc_f - acc_m),
    'z_test_p_value': float(p_value) if p_value is not None else None,
    'logit_unadj_coef_female': float(unadj_coef),
    'logit_unadj_or_female': float(unadj_or),
    'logit_unadj_p_value': float(unadj_p),
    'logit_adj_coef_female': float(adj_coef),
    'logit_adj_or_female': float(adj_or),
    'logit_adj_p_value': float(adj_p),
}

print(json.dumps(summary, indent=2))
