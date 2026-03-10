import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Column names based on info.json
female_col = 'feature2'   # 1 if female
accept_col = 'feature14'  # 1 if accepted

df = _df.copy()

# Basic checks
n_total = len(df)

# Drop rows with missing in key columns for unadjusted tests
key_cols = [female_col, accept_col]
_df_key = df.dropna(subset=key_cols)

# Unadjusted acceptance rates by gender
rates = _df_key.groupby(female_col)[accept_col].agg(['mean','count']).rename(index={0:'male',1:'female'})

# Chi-square test of independence
cont_table = pd.crosstab(_df_key[female_col], _df_key[accept_col])
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont_table)

# Logistic regression unadjusted
X_unadj = sm.add_constant(_df_key[[female_col]])
y = _df_key[accept_col]
logit_unadj = sm.Logit(y, X_unadj)
res_unadj = logit_unadj.fit(disp=False)

# Adjusted logistic regression (control for other features)
# Exclude outcome columns feature11 (denied) and feature14 (accepted)
exclude = {accept_col, 'feature11'}
control_cols = [c for c in df.columns if c not in exclude]

# Ensure female included (it is in control_cols)
_df_adj = df.dropna(subset=control_cols + [accept_col])
X_adj = sm.add_constant(_df_adj[control_cols])
y_adj = _df_adj[accept_col]

logit_adj = sm.Logit(y_adj, X_adj)
res_adj = logit_adj.fit(disp=False)

# Extract female effect
coef_unadj = res_unadj.params[female_col]
p_unadj = res_unadj.pvalues[female_col]

coef_adj = res_adj.params[female_col]
p_adj = res_adj.pvalues[female_col]

# Odds ratios
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# Summaries
summary = {
    'n_total': int(n_total),
    'n_unadjusted': int(len(_df_key)),
    'n_adjusted': int(len(_df_adj)),
    'accept_rate_male': float(rates.loc['male','mean']) if 'male' in rates.index else None,
    'accept_rate_female': float(rates.loc['female','mean']) if 'female' in rates.index else None,
    'count_male': int(rates.loc['male','count']) if 'male' in rates.index else None,
    'count_female': int(rates.loc['female','count']) if 'female' in rates.index else None,
    'chi2': float(chi2),
    'chi2_p': float(p_chi2),
    'unadjusted_or_female': or_unadj,
    'unadjusted_p_female': float(p_unadj),
    'adjusted_or_female': or_adj,
    'adjusted_p_female': float(p_adj),
}

print(json.dumps(summary, indent=2))
