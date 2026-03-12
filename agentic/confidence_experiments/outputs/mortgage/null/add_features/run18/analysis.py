import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

# Drop index-like column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Ensure numeric columns are numeric where expected
for col in df.columns:
    if df[col].dtype == 'object':
        # try coercing; keep as object if mostly non-numeric
        coerced = pd.to_numeric(df[col], errors='coerce')
        # if coercion yields a meaningful number of non-nulls, replace
        if coerced.notna().sum() > 0:
            # if originally object but mostly non-numeric, leave; else replace
            if coerced.notna().mean() > 0.8:
                df[col] = coerced

# Outcome
outcome = 'deny' if 'deny' in df.columns else 'accept'

# Basic checks
n_total = len(df)

# Female variable
if 'female' not in df.columns:
    raise ValueError('female column not found')

# Deny/accept relationship if both present
accept_deny_consistent = None
if 'accept' in df.columns and 'deny' in df.columns:
    accept_deny_consistent = ((df['accept'] + df['deny']) == 1).mean()

# Grouped denial rates
rate_by_female = df.groupby('female')[outcome].mean()
count_by_female = df['female'].value_counts().sort_index()

# Two-proportion z-test for denial rate difference
# group 1: female==1, group 0: female==0
female_mask = df['female'] == 1
male_mask = df['female'] == 0

# handle any missing outcomes
valid = df[outcome].notna() & df['female'].notna()
valid_female = valid & female_mask
valid_male = valid & male_mask

count_female = df.loc[valid_female, outcome].sum()
count_male = df.loc[valid_male, outcome].sum()

n_female = valid_female.sum()
n_male = valid_male.sum()

# z-test for proportions
z_stat, p_val = proportions_ztest([count_female, count_male], [n_female, n_male])

# Difference in denial rates (female - male)
rate_female = count_female / n_female
rate_male = count_male / n_male
rate_diff = rate_female - rate_male

# Approximate 95% CI for difference in proportions
se_diff = np.sqrt(rate_female*(1-rate_female)/n_female + rate_male*(1-rate_male)/n_male)
ci_low = rate_diff - 1.96*se_diff
ci_high = rate_diff + 1.96*se_diff

# Logistic regression unadjusted
logit_results = {}
X = sm.add_constant(df[['female']].astype(float))
y = df[outcome].astype(float)
mask = X.notna().all(axis=1) & y.notna()
model = sm.Logit(y[mask], X[mask])
res = model.fit(disp=0)
logit_results['unadjusted'] = {
    'coef_female': res.params['female'],
    'p_value': res.pvalues['female'],
    'odds_ratio': float(np.exp(res.params['female']))
}

# Adjusted model with available mortgage-related controls
controls = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
controls = [c for c in controls if c in df.columns]

X_adj = df[['female'] + controls].copy()
X_adj = X_adj.apply(pd.to_numeric, errors='coerce')
X_adj = sm.add_constant(X_adj)
mask_adj = X_adj.notna().all(axis=1) & y.notna()

adj_result = None
if mask_adj.sum() > 0:
    model_adj = sm.Logit(y[mask_adj], X_adj[mask_adj])
    res_adj = model_adj.fit(disp=0)
    adj_result = {
        'coef_female': res_adj.params['female'],
        'p_value': res_adj.pvalues['female'],
        'odds_ratio': float(np.exp(res_adj.params['female'])),
        'n_obs': int(mask_adj.sum())
    }

summary = {
    'n_total': int(n_total),
    'accept_deny_consistent': accept_deny_consistent,
    'rate_by_female': rate_by_female.to_dict(),
    'count_by_female': count_by_female.to_dict(),
    'z_test_p_value': float(p_val),
    'rate_diff_female_minus_male': float(rate_diff),
    'rate_diff_ci_95': [float(ci_low), float(ci_high)],
    'logit_unadjusted': logit_results['unadjusted'],
    'logit_adjusted': adj_result,
}

print(json.dumps(summary, indent=2))
