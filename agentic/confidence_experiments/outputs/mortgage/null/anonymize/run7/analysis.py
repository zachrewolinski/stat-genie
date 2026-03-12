import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan)

# Define columns
female_col = "feature2"  # 1 if applicant is female, 0 if male
accept_col = "feature14"  # 1 if accepted, 0 if denied

# Basic checks
n_total = len(df)

# Approval rates by gender
rates = df.groupby(female_col)[accept_col].mean()
counts = df.groupby(female_col)[accept_col].agg(['count', 'sum'])

# Difference in approval rates (female - male)
approval_diff = rates.get(1, np.nan) - rates.get(0, np.nan)

# 2x2 contingency for chi-square
contingency = pd.crosstab(df[female_col], df[accept_col])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression
df_unadj = df[[accept_col, female_col]].dropna()
X_unadj = sm.add_constant(df_unadj[[female_col]])
logit_unadj = sm.Logit(df_unadj[accept_col], X_unadj).fit(disp=False)

# Adjusted logistic regression (control for other features except acceptance/denial and gender)
feature_cols = [c for c in df.columns if c not in {accept_col, "feature11", female_col}]
df_adj = df[[accept_col, female_col] + feature_cols].dropna()
X_adj = sm.add_constant(df_adj[[female_col] + feature_cols])
logit_adj = sm.Logit(df_adj[accept_col], X_adj).fit(disp=False)

# Extract effect sizes
unadj_coef = logit_unadj.params[female_col]
unadj_p = logit_unadj.pvalues[female_col]
unadj_or = np.exp(unadj_coef)

adj_coef = logit_adj.params[female_col]
adj_p = logit_adj.pvalues[female_col]
adj_or = np.exp(adj_coef)

# Prepare summary
summary = {
    "n_total": int(n_total),
    "approval_rate_female": float(rates.get(1, np.nan)),
    "approval_rate_male": float(rates.get(0, np.nan)),
    "approval_diff_female_minus_male": float(approval_diff),
    "chi2_p": float(p_chi),
    "unadj_or": float(unadj_or),
    "unadj_p": float(unadj_p),
    "adj_or": float(adj_or),
    "adj_p": float(adj_p),
}

print(json.dumps(summary, indent=2))
