import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Core variables
# accept: 1 accepted, 0 denied
# female: 1 female, 0 male

# Drop rows with missing key vars
key_cols = ['accept', 'female']
base = df[key_cols].dropna()

# Contingency table and chi-square test
ct = pd.crosstab(base['female'], base['accept'])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Compute approval rates
approval_rates = base.groupby('female')['accept'].mean()

# Logistic regression (unadjusted)
X_unadj = sm.add_constant(base[['female']])
model_unadj = sm.Logit(base['accept'], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Logistic regression (adjusted for common creditworthiness and application features)
controls = [
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

# Use only rows with complete data for model
model_cols = ['accept', 'female'] + controls
model_df = df[model_cols].dropna()

X_adj = sm.add_constant(model_df[['female'] + controls])
model_adj = sm.Logit(model_df['accept'], X_adj)
res_adj = model_adj.fit(disp=False)

# Extract effect for female
coef_unadj = res_unadj.params['female']
se_unadj = res_unadj.bse['female']
OR_unadj = float(np.exp(coef_unadj))

coef_adj = res_adj.params['female']
se_adj = res_adj.bse['female']
OR_adj = float(np.exp(coef_adj))

# Wald p-values
p_unadj = res_unadj.pvalues['female']
p_adj = res_adj.pvalues['female']

results = {
    'n_total': int(len(df)),
    'n_base': int(len(base)),
    'approval_rate_male': float(approval_rates.get(0.0, np.nan)),
    'approval_rate_female': float(approval_rates.get(1.0, np.nan)),
    'chi2_p': float(p_chi),
    'unadjusted': {
        'coef': float(coef_unadj),
        'se': float(se_unadj),
        'p': float(p_unadj),
        'OR': OR_unadj
    },
    'adjusted': {
        'coef': float(coef_adj),
        'se': float(se_adj),
        'p': float(p_adj),
        'OR': OR_adj,
        'n_model': int(len(model_df))
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
