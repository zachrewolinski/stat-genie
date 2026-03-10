import json
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

DATA_PATH = 'mortgage.csv'

df = pd.read_csv(DATA_PATH)

# Basic checks
cols = df.columns.tolist()

# Identify outcome: accept or deny
# Use 'accept' if exists; else use 'deny' (1 denied) and invert
if 'accept' in df.columns:
    y = df['accept']
    outcome_name = 'accept'
    if y.dropna().isin([0,1]).all():
        pass
    else:
        # if accept not binary, try deny
        y = None
if 'accept' not in df.columns or y is None:
    if 'deny' in df.columns:
        y = 1 - df['deny']
        outcome_name = 'accept_from_deny'
    else:
        raise ValueError('No accept/deny column found')

# Gender variable
if 'female' in df.columns:
    female = df['female']
else:
    raise ValueError('No female column found')

# Clean rows with missing data in outcome or female
mask = female.notna() & y.notna()
sub = df.loc[mask].copy()
sub['accept'] = y.loc[mask]
sub['female'] = female.loc[mask]

# Ensure binary
sub = sub[sub['female'].isin([0,1]) & sub['accept'].isin([0,1])]

# Two-proportion z-test
counts = sub.groupby('female')['accept'].sum()
ns = sub.groupby('female')['accept'].count()
# female=1 vs female=0
count = np.array([counts.get(1,0), counts.get(0,0)])
obs = np.array([ns.get(1,0), ns.get(0,0)])
stat, pval = proportions_ztest(count, obs)

# Effect size: difference in acceptance rates
rate_f = counts.get(1,0) / ns.get(1,0)
rate_m = counts.get(0,0) / ns.get(0,0)
rate_diff = rate_f - rate_m

# Logistic regression: accept ~ female + key controls if available
control_vars = [
    'black','housing_expense_ratio','self_employed','married','mortgage_credit',
    'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI','age','occupation'
]
controls = [c for c in control_vars if c in sub.columns]

logit_res = None
if controls:
    X = sub[['female'] + controls]
    # Drop rows with missing in controls
    X = X.dropna()
    y_aligned = sub.loc[X.index, 'accept']
    X = sm.add_constant(X)
    try:
        logit_model = sm.Logit(y_aligned, X)
        logit_res = logit_model.fit(disp=False)
    except Exception:
        logit_res = None

output = {
    'n_total': int(len(sub)),
    'accept_rate_female': float(rate_f),
    'accept_rate_male': float(rate_m),
    'rate_diff_f_minus_m': float(rate_diff),
    'ztest_stat': float(stat),
    'ztest_pval': float(pval),
    'controls_used': controls,
}

if logit_res is not None:
    coef = logit_res.params['female']
    se = logit_res.bse['female']
    p = logit_res.pvalues['female']
    odds_ratio = float(np.exp(coef))
    output.update({
        'logit_coef_female': float(coef),
        'logit_se_female': float(se),
        'logit_pval_female': float(p),
        'logit_odds_ratio_female': odds_ratio,
        'logit_n': int(logit_res.nobs),
    })

print(json.dumps(output, indent=2))
