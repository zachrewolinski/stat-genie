import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
DF = pd.read_csv('mortgage.csv')

# Ensure key columns exist
cols = DF.columns

# Outcome: deny (1 denied). Gender: female (1 female)
# Clean: drop rows with missing in relevant columns
base_cols = ['deny', 'female']

# Covariates likely relevant for mortgage approval
covariates = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI',
]

use_covariates = [c for c in covariates if c in cols]

# Data for unadjusted
df_unadj = DF[base_cols].dropna().copy()

# Two-proportion test / chi-square
ct = pd.crosstab(df_unadj['female'], df_unadj['deny'])
# ensure both columns 0/1 exist
for v in [0, 1]:
    if v not in ct.columns:
        ct[v] = 0
ct = ct[[0, 1]]

chi2, p_chi2, dof, expected = stats.chi2_contingency(ct.values)

# Unadjusted logistic regression
X_unadj = sm.add_constant(df_unadj['female'])
model_unadj = sm.Logit(df_unadj['deny'], X_unadj).fit(disp=False)

# Adjusted logistic regression
adj_cols = ['deny', 'female'] + use_covariates
df_adj = DF[adj_cols].dropna().copy()
X_adj = df_adj[['female'] + use_covariates]
X_adj = sm.add_constant(X_adj)
model_adj = sm.Logit(df_adj['deny'], X_adj).fit(disp=False)

# Effect sizes
def odds_ratio_ci(model, var):
    coef = model.params[var]
    se = model.bse[var]
    or_ = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    p = model.pvalues[var]
    return or_, ci_low, ci_high, p

or_unadj, ci_l_u, ci_h_u, p_unadj = odds_ratio_ci(model_unadj, 'female')
or_adj, ci_l_a, ci_h_a, p_adj = odds_ratio_ci(model_adj, 'female')

# Denial rates
rate_female = df_unadj.loc[df_unadj['female'] == 1, 'deny'].mean()
rate_male = df_unadj.loc[df_unadj['female'] == 0, 'deny'].mean()

# Sample sizes
n_total = len(df_unadj)

summary = {
    'n_total': n_total,
    'n_female': int((df_unadj['female'] == 1).sum()),
    'n_male': int((df_unadj['female'] == 0).sum()),
    'deny_rate_female': float(rate_female),
    'deny_rate_male': float(rate_male),
    'chi2_p': float(p_chi2),
    'unadj_or': float(or_unadj),
    'unadj_or_ci': [float(ci_l_u), float(ci_h_u)],
    'unadj_p': float(p_unadj),
    'adj_or': float(or_adj),
    'adj_or_ci': [float(ci_l_a), float(ci_h_a)],
    'adj_p': float(p_adj),
    'adj_n': int(len(df_adj)),
}

print(summary)
