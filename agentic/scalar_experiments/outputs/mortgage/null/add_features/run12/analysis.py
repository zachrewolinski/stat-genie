import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Core mortgage-related variables
outcome = 'accept'
key = 'female'
controls = [
    'black',
    'married',
    'self_employed',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'housing_expense_ratio',
    'PI_ratio',
    'loan_to_value',
]

# Keep relevant columns, drop missing
cols = [outcome, key] + controls
sub = df[cols].copy()
sub = sub.dropna()

# Basic counts
n_total = len(sub)
counts = sub[key].value_counts().to_dict()

# Approval rates by gender
rate_by_gender = sub.groupby(key)[outcome].mean().to_dict()

# Two-proportion z test (female=1 vs male=0)
# Ensure order: male (0), female (1)
counts_accept = sub.groupby(key)[outcome].sum().reindex([0.0, 1.0]).values
n_by_gender = sub.groupby(key)[outcome].count().reindex([0.0, 1.0]).values

# Convert to int arrays
counts_accept = counts_accept.astype(int)
n_by_gender = n_by_gender.astype(int)

z_stat, p_value = proportions_ztest(count=counts_accept, nobs=n_by_gender)

# Difference in approval rates (female - male)
rate_male = rate_by_gender.get(0.0, np.nan)
rate_female = rate_by_gender.get(1.0, np.nan)
rate_diff = rate_female - rate_male

# Logistic regression: unadjusted
X_unadj = sm.add_constant(sub[[key]])
model_unadj = sm.Logit(sub[outcome], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Logistic regression: adjusted
X_adj = sm.add_constant(sub[[key] + controls])
model_adj = sm.Logit(sub[outcome], X_adj)
res_adj = model_adj.fit(disp=False, maxiter=200)

# Extract female effect
coef_unadj = res_unadj.params[key]
p_unadj = res_unadj.pvalues[key]

coef_adj = res_adj.params[key]
p_adj = res_adj.pvalues[key]

odds_unadj = float(np.exp(coef_unadj))
odds_adj = float(np.exp(coef_adj))

summary = {
    'n_total': int(n_total),
    'counts_by_gender': {str(k): int(v) for k, v in counts.items()},
    'approval_rate_male': float(rate_male),
    'approval_rate_female': float(rate_female),
    'rate_diff_female_minus_male': float(rate_diff),
    'proportions_ztest': {
        'z_stat': float(z_stat),
        'p_value': float(p_value),
    },
    'logit_unadjusted': {
        'coef_female': float(coef_unadj),
        'p_value_female': float(p_unadj),
        'odds_ratio_female': float(odds_unadj),
    },
    'logit_adjusted': {
        'coef_female': float(coef_adj),
        'p_value_female': float(p_adj),
        'odds_ratio_female': float(odds_adj),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
