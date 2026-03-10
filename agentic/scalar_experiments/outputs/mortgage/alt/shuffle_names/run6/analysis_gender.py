import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# According to info.json descriptions, denied_PMI is female indicator (1=female),
# deny is acceptance (1=accepted), self_employed is denial.
# We'll use denied_PMI as gender and deny as approval.

df = _df.copy()

# Use columns
gender_col = 'denied_PMI'
approval_col = 'deny'

# Drop missing in key vars
sub = df[[gender_col, approval_col]].dropna()

# Ensure binary
for col in [gender_col, approval_col]:
    # coerce to int 0/1
    sub[col] = sub[col].astype(int)

n = len(sub)

# Counts
female = sub[sub[gender_col] == 1]
male = sub[sub[gender_col] == 0]

n_f = len(female)
n_m = len(male)

# Approval rates
rate_f = female[approval_col].mean()
rate_m = male[approval_col].mean()

# Difference in proportions test (two-proportion z-test)
# Use pooled proportion
p_pool = (female[approval_col].sum() + male[approval_col].sum()) / n
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_f + 1 / n_m))
if se > 0:
    z = (rate_f - rate_m) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_value = np.nan

# 95% CI for difference (Wald)
se_diff = np.sqrt(rate_f * (1 - rate_f) / n_f + rate_m * (1 - rate_m) / n_m)
ci_low = (rate_f - rate_m) - 1.96 * se_diff
ci_high = (rate_f - rate_m) + 1.96 * se_diff

# Odds ratio (unadjusted)
# add 0.5 to avoid zero counts if needed
f_accept = female[approval_col].sum()
f_deny = n_f - f_accept
m_accept = male[approval_col].sum()
m_deny = n_m - m_accept

# Haldane-Anscombe correction if any zero
if min(f_accept, f_deny, m_accept, m_deny) == 0:
    f_accept += 0.5
    f_deny += 0.5
    m_accept += 0.5
    m_deny += 0.5

odds_ratio = (f_accept / f_deny) / (m_accept / m_deny)

# Logistic regression (unadjusted)
X = sm.add_constant(sub[[gender_col]])
model = sm.Logit(sub[approval_col], X).fit(disp=False)
coef = model.params[gender_col]
se_coef = model.bse[gender_col]

# 95% CI for odds ratio
or_ci_low = np.exp(coef - 1.96 * se_coef)
or_ci_high = np.exp(coef + 1.96 * se_coef)

# Adjusted model with other covariates based on info.json descriptions
# Exclude columns likely to be IDs or outcome complements
covariates = [
    'consumer_credit',        # described as black
    'mortgage_credit',        # housing expense ratio
    'accept',                 # self-employed
    'loan_to_value',          # married
    'married',                # mortgage credit score (1-4)
    'black',                  # consumer credit score (1-6)
    'PI_ratio',               # bad credit history
    'housing_expense_ratio',  # debt ratio
    'Unnamed: 0',             # loan-to-value ratio
]

adj_cols = [gender_col, approval_col] + covariates
adj = df[adj_cols].dropna()

# Build model
X_adj = sm.add_constant(adj[[gender_col] + covariates])
model_adj = sm.Logit(adj[approval_col], X_adj).fit(disp=False)
coef_adj = model_adj.params[gender_col]
se_adj = model_adj.bse[gender_col]

or_adj = np.exp(coef_adj)
or_adj_ci_low = np.exp(coef_adj - 1.96 * se_adj)
or_adj_ci_high = np.exp(coef_adj + 1.96 * se_adj)

p_adj = model_adj.pvalues[gender_col]

# Output results
print('N total', n)
print('N female', n_f, 'N male', n_m)
print('approval rate female', rate_f)
print('approval rate male', rate_m)
print('diff female-male', rate_f - rate_m)
print('diff 95% CI', ci_low, ci_high)
print('two-proportion z', z, 'p', p_value)
print('unadj odds ratio', odds_ratio)
print('logit coef', coef, 'p', model.pvalues[gender_col])
print('logit OR 95% CI', or_ci_low, or_ci_high)
print('Adjusted logit OR', or_adj, '95% CI', or_adj_ci_low, or_adj_ci_high, 'p', p_adj)
print('Adjusted N', len(adj))

