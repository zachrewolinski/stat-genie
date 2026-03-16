import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv('mortgage.csv')

# Identify columns (based on metadata and prevalence)
# "denied_PMI" column appears to be the gender indicator (female=1)
# "deny" column appears to be approval (mean ~0.88)

gender_col = 'denied_PMI'
approval_col = 'deny'

# Drop missing for key columns
_df = _df.dropna(subset=[gender_col, approval_col])

# Basic rates by gender
rates = _df.groupby(gender_col)[approval_col].mean()
counts = _df.groupby(gender_col)[approval_col].count()

# Two-proportion z-test (female=1 vs male=0)
count = np.array([
    _df.loc[_df[gender_col] == 1, approval_col].sum(),
    _df.loc[_df[gender_col] == 0, approval_col].sum(),
])
obs = np.array([
    (_df[gender_col] == 1).sum(),
    (_df[gender_col] == 0).sum(),
])
stat, pval = proportions_ztest(count, obs)

# Unadjusted logistic regression
X_unadj = sm.add_constant(_df[[gender_col]])
y = _df[approval_col]
model_unadj = sm.Logit(y, X_unadj).fit(disp=False)

# Adjusted logistic regression with other covariates
# Include gender plus all other covariates (excluding outcome)
covariates = [c for c in _df.columns if c != approval_col]
X_adj = _df[covariates]
# Drop rows with missing in covariates
mask = X_adj.notna().all(axis=1)
X_adj = X_adj[mask]
y_adj = y[mask]
X_adj = sm.add_constant(X_adj)
model_adj = sm.Logit(y_adj, X_adj).fit(disp=False, maxiter=200)

# Extract gender coefficient
coef_unadj = model_unadj.params[gender_col]
p_unadj = model_unadj.pvalues[gender_col]
or_unadj = np.exp(coef_unadj)

coef_adj = model_adj.params[gender_col]
p_adj = model_adj.pvalues[gender_col]
or_adj = np.exp(coef_adj)

print('Approval rates by gender (female=1):')
print(rates)
print('Counts:')
print(counts)
print('\nTwo-proportion z-test p-value:', pval)
print('\nUnadjusted logit coef:', coef_unadj, 'OR:', or_unadj, 'p:', p_unadj)
print('Adjusted logit coef:', coef_adj, 'OR:', or_adj, 'p:', p_adj)
