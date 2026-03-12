import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Define columns
gender_col = "feature2"  # 1 if female, 0 if male
approval_col = "feature14"  # 1 if accepted, 0 if denied

# Basic checks
# Drop rows with missing in key columns
key_cols = [gender_col, approval_col]

# Controls: all other features except approval and denial
control_cols = [
    "feature1",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature12",
    "feature13",
]

cols_needed = key_cols + control_cols

df_clean = df[cols_needed].dropna().copy()

# Unadjusted approval rates
rate_by_gender = df_clean.groupby(gender_col)[approval_col].mean()
count_by_gender = df_clean.groupby(gender_col)[approval_col].count()

# 2x2 contingency table for chi-square
ct = pd.crosstab(df_clean[gender_col], df_clean[approval_col])
# Ensure columns 0/1 order
ct = ct.reindex(index=[0,1], columns=[0,1], fill_value=0)
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

# Unadjusted logistic regression
X_unadj = sm.add_constant(df_clean[[gender_col]])
y = df_clean[approval_col]
logit_unadj = sm.Logit(y, X_unadj).fit(disp=False)
coef_unadj = logit_unadj.params[gender_col]
se_unadj = logit_unadj.bse[gender_col]
p_unadj = logit_unadj.pvalues[gender_col]
# Odds ratio and 95% CI
or_unadj = float(np.exp(coef_unadj))
ci_unadj = np.exp(coef_unadj + np.array([-1, 1]) * 1.96 * se_unadj)

# Adjusted logistic regression
X_adj = sm.add_constant(df_clean[[gender_col] + control_cols])
logit_adj = sm.Logit(y, X_adj).fit(disp=False)
coef_adj = logit_adj.params[gender_col]
se_adj = logit_adj.bse[gender_col]
p_adj = logit_adj.pvalues[gender_col]
or_adj = float(np.exp(coef_adj))
ci_adj = np.exp(coef_adj + np.array([-1, 1]) * 1.96 * se_adj)

results = {
    "n": int(len(df_clean)),
    "approval_rate_male": float(rate_by_gender.get(0, np.nan)),
    "approval_rate_female": float(rate_by_gender.get(1, np.nan)),
    "count_male": int(count_by_gender.get(0, 0)),
    "count_female": int(count_by_gender.get(1, 0)),
    "chi2": float(chi2),
    "chi2_p": float(p_chi2),
    "unadj_or": or_unadj,
    "unadj_or_ci": [float(ci_unadj[0]), float(ci_unadj[1])],
    "unadj_p": float(p_unadj),
    "adj_or": or_adj,
    "adj_or_ci": [float(ci_adj[0]), float(ci_adj[1])],
    "adj_p": float(p_adj),
}

print(json.dumps(results, indent=2))
