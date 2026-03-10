import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm

# Load data
df = pd.read_csv('mortgage.csv')

# Ensure binary columns are numeric 0/1
# Target: accept (1=approved) or deny (1=denied)

# Basic counts
n_total = len(df)

# Drop rows with missing in relevant columns
cols_base = ['accept', 'female']

# For adjusted model, include key pre-approval covariates
covariates = [
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
]

# Unadjusted contingency
cont_df = df[cols_base].dropna()
contingency = pd.crosstab(cont_df['female'], cont_df['accept'])

# Ensure both categories present
chi2 = None
p_chi2 = None
if contingency.shape == (2,2):
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)

# Acceptance rates
rates = cont_df.groupby('female')['accept'].mean()
rate_female = rates.get(1, np.nan)
rate_male = rates.get(0, np.nan)
rate_diff = rate_female - rate_male

# Unadjusted logit: accept ~ female
unadj_df = df[['accept', 'female']].dropna()
X_unadj = sm.add_constant(unadj_df['female'])
model_unadj = sm.Logit(unadj_df['accept'], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Adjusted logit
adj_df = df[['accept'] + covariates].dropna()
X_adj = sm.add_constant(adj_df[covariates])
model_adj = sm.Logit(adj_df['accept'], X_adj)
res_adj = model_adj.fit(disp=False)

# Extract female effect
coef_unadj = res_unadj.params['female']
p_unadj = res_unadj.pvalues['female']

coef_adj = res_adj.params['female']
p_adj = res_adj.pvalues['female']

# Odds ratios
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# Marginal effect (approx at mean using discrete change)
# Use statsmodels get_margeff if available
try:
    margeff = res_adj.get_margeff(at='mean', method='dydx')
    me_female = float(margeff.margeff[margeff.params.index.get_loc('female')])
except Exception:
    me_female = None

# Output summary
summary = {
    'n_total': int(n_total),
    'n_unadj': int(len(unadj_df)),
    'n_adj': int(len(adj_df)),
    'accept_rate_female': float(rate_female),
    'accept_rate_male': float(rate_male),
    'accept_rate_diff': float(rate_diff),
    'chi2_pvalue': None if p_chi2 is None else float(p_chi2),
    'unadj_coef_female': float(coef_unadj),
    'unadj_or_female': float(or_unadj),
    'unadj_p_female': float(p_unadj),
    'adj_coef_female': float(coef_adj),
    'adj_or_female': float(or_adj),
    'adj_p_female': float(p_adj),
    'adj_marginal_effect_female': None if me_female is None else float(me_female),
}

print(summary)
