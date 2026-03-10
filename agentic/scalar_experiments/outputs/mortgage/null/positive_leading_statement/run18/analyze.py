import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop unnamed index if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure binary columns are numeric
for col in ["female", "accept", "deny", "black", "self_employed", "married", "bad_history", "denied_PMI"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic counts
n_total = len(df)

# Approval rates by gender
rate_by_gender = df.groupby("female")["accept"].mean()
counts_by_gender = df["female"].value_counts(dropna=False).to_dict()

# Contingency table for chi-square
cont_table = pd.crosstab(df["female"], df["accept"])
chi2, chi_p, dof, expected = stats.chi2_contingency(cont_table)

# Logistic regression: unadjusted
X_unadj = sm.add_constant(df[["female"]])
model_unadj = sm.Logit(df["accept"], X_unadj, missing="drop")
res_unadj = model_unadj.fit(disp=False)

# Adjusted model with controls
controls = [
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]
controls = [c for c in controls if c in df.columns]
X_adj = sm.add_constant(df[["female"] + controls])
model_adj = sm.Logit(df["accept"], X_adj, missing="drop")
res_adj = model_adj.fit(disp=False)

# Extract effect sizes

def odds_ratio_and_ci(res, var):
    coef = res.params[var]
    se = res.bse[var]
    or_val = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    pval = float(res.pvalues[var])
    return {
        "odds_ratio": or_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": pval,
    }

unadj_stats = odds_ratio_and_ci(res_unadj, "female")
adj_stats = odds_ratio_and_ci(res_adj, "female")

summary = {
    "n_total": int(n_total),
    "counts_by_gender": {str(k): int(v) for k, v in counts_by_gender.items()},
    "approval_rate_female_0": float(rate_by_gender.get(0, np.nan)),
    "approval_rate_female_1": float(rate_by_gender.get(1, np.nan)),
    "chi2_p_value": float(chi_p),
    "unadjusted": unadj_stats,
    "adjusted": adj_stats,
}

print(json.dumps(summary, indent=2))
