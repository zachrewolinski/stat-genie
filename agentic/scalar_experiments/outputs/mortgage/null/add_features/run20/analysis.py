import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic columns
outcome = 'accept'
key = 'female'

# Ensure binary outcomes
# acceptance rates by gender
summary = df[[key, outcome]].dropna()

# Crosstab and chi-square
ct = pd.crosstab(summary[key], summary[outcome])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Difference in acceptance rates
rates = summary.groupby(key)[outcome].mean()
rate_diff = rates.loc[1] - rates.loc[0]

# Two-proportion z-test
n1 = summary[summary[key]==1][outcome].count()
n0 = summary[summary[key]==0][outcome].count()

p1 = rates.loc[1]
p0 = rates.loc[0]
p_pool = (p1*n1 + p0*n0) / (n1 + n0)
se_pool = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n0))
z = (p1 - p0) / se_pool if se_pool > 0 else np.nan
p_z = 2*(1-stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Logistic regression - unadjusted
logit_df = summary.copy()
X = sm.add_constant(logit_df[[key]])
model_unadj = sm.Logit(logit_df[outcome], X).fit(disp=False)

# Logistic regression - adjusted with relevant covariates
covariates = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value'
]

# Keep only available columns
covariates = [c for c in covariates if c in df.columns]
logit_adj_df = df[[outcome, key] + covariates].dropna()
X_adj = sm.add_constant(logit_adj_df[[key] + covariates])
model_adj = sm.Logit(logit_adj_df[outcome], X_adj).fit(disp=False)

# Extract coefficient and p-values
coef_unadj = model_unadj.params[key]
p_unadj = model_unadj.pvalues[key]

coef_adj = model_adj.params[key]
p_adj = model_adj.pvalues[key]

# Convert to odds ratios for interpretation
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# Confidence intervals (95%) for adjusted
conf_adj = model_adj.conf_int().loc[key]
or_adj_ci = np.exp(conf_adj).values

# Output results
print('N total:', len(df))
print('N used unadjusted:', len(logit_df))
print('N used adjusted:', len(logit_adj_df))
print('Acceptance rates by gender (0=male,1=female):')
print(rates)
print('Rate diff (female - male):', rate_diff)
print('Chi-square p:', p_chi)
print('Two-proportion z p:', p_z)
print('Unadjusted logit coef:', coef_unadj, 'p:', p_unadj, 'OR:', or_unadj)
print('Adjusted logit coef:', coef_adj, 'p:', p_adj, 'OR:', or_adj)
print('Adjusted OR 95% CI:', or_adj_ci)
