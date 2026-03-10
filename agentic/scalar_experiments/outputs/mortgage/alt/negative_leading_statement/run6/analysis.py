import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv('mortgage.csv')

# Basic cleanup
# Ensure numeric types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define outcome: accept (1 accepted, 0 denied)
# If accept missing but deny present, we could infer; assume accept present.

# Drop rows with missing values in key columns
outcome = 'accept'
key_cols = ['female', outcome]

# Additional covariates for adjusted model
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
    'denied_PMI'
]

# Keep rows with non-missing for outcome and female
base_df = df[key_cols].dropna()

# Unadjusted acceptance rates
rates = base_df.groupby('female')[outcome].mean()
counts = base_df.groupby('female')[outcome].agg(['count','sum'])

# Chi-square test of independence
contingency = pd.crosstab(base_df['female'], base_df[outcome])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression
X_unadj = sm.add_constant(base_df['female'])
model_unadj = sm.Logit(base_df[outcome], X_unadj).fit(disp=False)

# Adjusted logistic regression
adj_cols = ['female'] + covariates
adj_df = df[adj_cols + [outcome]].dropna()
X_adj = sm.add_constant(adj_df[adj_cols])
model_adj = sm.Logit(adj_df[outcome], X_adj).fit(disp=False)

# Extract female effect
coef_unadj = model_unadj.params['female']
se_unadj = model_unadj.bse['female']
p_unadj = model_unadj.pvalues['female']

coef_adj = model_adj.params['female']
se_adj = model_adj.bse['female']
p_adj = model_adj.pvalues['female']

# Odds ratios and 95% CI
or_unadj = float(np.exp(coef_unadj))
ci_unadj = np.exp(model_unadj.conf_int().loc['female'])

or_adj = float(np.exp(coef_adj))
ci_adj = np.exp(model_adj.conf_int().loc['female'])

results = {
    'n_total': int(len(df)),
    'n_base': int(len(base_df)),
    'n_adj': int(len(adj_df)),
    'accept_rate_male': float(rates.get(0, np.nan)),
    'accept_rate_female': float(rates.get(1, np.nan)),
    'counts': {int(k): {'count': int(v['count']), 'accepted': int(v['sum'])} for k, v in counts.iterrows()},
    'chi2': float(chi2),
    'p_chi2': float(p_chi2),
    'unadj': {
        'coef': float(coef_unadj),
        'se': float(se_unadj),
        'p': float(p_unadj),
        'or': float(or_unadj),
        'ci_low': float(ci_unadj[0]),
        'ci_high': float(ci_unadj[1]),
    },
    'adj': {
        'coef': float(coef_adj),
        'se': float(se_adj),
        'p': float(p_adj),
        'or': float(or_adj),
        'ci_low': float(ci_adj[0]),
        'ci_high': float(ci_adj[1]),
    },
}

print(json.dumps(results, indent=2))
