import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Standardize column names just in case
# (but keep originals)

df = _df.copy()

# Ensure required columns exist
required_cols = [
    "female", "accept", "deny", "black", "housing_expense_ratio", "self_employed",
    "married", "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
    "loan_to_value", "denied_PMI"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Basic summary
n_total = len(df)

# Drop rows with missing in key columns for tests
key_cols = ["female", "accept", "deny"]
chi_df = df[key_cols].dropna().copy()

# 2x2 contingency for female vs accept
contingency = pd.crosstab(chi_df["female"], chi_df["accept"])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Approval rates by gender
rates = chi_df.groupby("female")["accept"].mean()
counts = chi_df.groupby("female")["accept"].count()

# Unadjusted logistic regression: accept ~ female (GLM binomial)
logit_df = df[["accept", "female"]].dropna().copy()
X_unadj = sm.add_constant(logit_df["female"])
model_unadj = sm.GLM(logit_df["accept"], X_unadj, family=sm.families.Binomial()).fit()

# Adjusted logistic regression with controls (GLM binomial)
control_cols = [
    "female", "black", "housing_expense_ratio", "self_employed", "married",
    "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
    "loan_to_value", "denied_PMI"
]
logit_df_adj = df[["accept"] + control_cols].dropna().copy()
X_adj = sm.add_constant(logit_df_adj[control_cols])
model_adj = sm.GLM(logit_df_adj["accept"], X_adj, family=sm.families.Binomial()).fit(cov_type="HC3")

# Extract female coefficient and p-value
coef_unadj = model_unadj.params["female"]
p_unadj = model_unadj.pvalues["female"]

coef_adj = model_adj.params["female"]
p_adj = model_adj.pvalues["female"]

# Odds ratio and 95% CI for adjusted model
se_adj = model_adj.bse["female"]
ci_low = coef_adj - 1.96 * se_adj
ci_high = coef_adj + 1.96 * se_adj
or_adj = float(np.exp(coef_adj))
or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

# Build results dict
results = {
    "n_total": int(n_total),
    "n_chi": int(len(chi_df)),
    "approval_rate_female_0": float(rates.get(0, np.nan)),
    "approval_rate_female_1": float(rates.get(1, np.nan)),
    "count_female_0": int(counts.get(0, 0)),
    "count_female_1": int(counts.get(1, 0)),
    "chi2": float(chi2),
    "chi2_p": float(p_chi),
    "unadj_logit_coef": float(coef_unadj),
    "unadj_logit_p": float(p_unadj),
    "adj_logit_coef": float(coef_adj),
    "adj_logit_p": float(p_adj),
    "adj_odds_ratio": or_adj,
    "adj_or_ci_low": or_ci[0],
    "adj_or_ci_high": or_ci[1],
}

print(json.dumps(results, indent=2))
