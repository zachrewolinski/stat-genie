import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'mortgage.csv'

df = pd.read_csv(DATA_PATH)

# Basic cleanup: ensure binary columns are numeric
binary_cols = ['female','deny','accept','black','self_employed','married','bad_history','denied_PMI']
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Outcome: deny (1=denied)
# If accept exists but deny missing, derive deny
if 'deny' not in df.columns and 'accept' in df.columns:
    df['deny'] = 1 - df['accept']

# Compute unadjusted denial rates by gender
rate_table = df.groupby('female')['deny'].agg(['mean','count'])
rate_table['deny_rate'] = rate_table['mean']
rate_table = rate_table[['count','deny_rate']]

# Chi-square test for independence on female vs deny
contingency = pd.crosstab(df['female'], df['deny'])
chi2, p_chi, dof, exp = stats.chi2_contingency(contingency)

# Logistic regression unadjusted
unadj_cols = ['female']
model_df = df[['deny'] + unadj_cols].dropna()
X_unadj = sm.add_constant(model_df[unadj_cols])
model_unadj = sm.Logit(model_df['deny'], X_unadj).fit(disp=False)

# Adjusted model with mortgage-relevant covariates
covariates = [
    'female','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio',
    'loan_to_value','denied_PMI'
]
# Keep only available columns
covariates = [c for c in covariates if c in df.columns]
model_df_adj = df[['deny'] + covariates].dropna()
X_adj = sm.add_constant(model_df_adj[covariates])
model_adj = sm.Logit(model_df_adj['deny'], X_adj).fit(disp=False)

# Extract odds ratios and p-values for female
unadj_or = np.exp(model_unadj.params['female'])
unadj_p = model_unadj.pvalues['female']
adj_or = np.exp(model_adj.params['female'])
adj_p = model_adj.pvalues['female']

# Also compute marginal difference in denial rates
deny_rate_female = rate_table.loc[1, 'deny_rate'] if 1 in rate_table.index else np.nan
deny_rate_male = rate_table.loc[0, 'deny_rate'] if 0 in rate_table.index else np.nan
rate_diff = deny_rate_female - deny_rate_male

# Build summary dict
summary = {
    'n_total': int(df.shape[0]),
    'n_used_unadj': int(model_df.shape[0]),
    'n_used_adj': int(model_df_adj.shape[0]),
    'deny_rate_female': float(deny_rate_female),
    'deny_rate_male': float(deny_rate_male),
    'rate_diff_female_minus_male': float(rate_diff),
    'chi2_pvalue': float(p_chi),
    'unadj_or_female': float(unadj_or),
    'unadj_p_female': float(unadj_p),
    'adj_or_female': float(adj_or),
    'adj_p_female': float(adj_p),
}

print('RATE_TABLE')
print(rate_table)
print('\nCHI2')
print({'chi2': chi2, 'p': p_chi, 'dof': dof})
print('\nUNADJ_LOGIT')
print(model_unadj.summary())
print('\nADJ_LOGIT')
print(model_adj.summary())
print('\nSUMMARY')
print(summary)
