import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop unnamed index-like column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure binary columns are numeric
binary_cols = ["female", "accept", "deny", "black", "self_employed", "married", "bad_history", "denied_PMI"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Outcome: acceptance
if "accept" in df.columns:
    outcome = "accept"
else:
    outcome = "deny"

# Basic group stats
# For accept: mean is acceptance rate; for deny: mean is denial rate
rate_by_gender = df.groupby("female")[outcome].mean()
count_by_gender = df.groupby("female")[outcome].count()

# 2x2 contingency table: female vs outcome (accept)
if outcome == "accept":
    table = pd.crosstab(df["female"], df["accept"])
else:
    # If outcome is deny, also compute for accept if available
    table = pd.crosstab(df["female"], df[outcome])

chi2, chi2_p, _, _ = stats.chi2_contingency(table)

# Unadjusted logistic regression: accept ~ female
if outcome == "accept":
    unadj_model = smf.logit("accept ~ female", data=df).fit(disp=False)
    unadj_coef = unadj_model.params["female"]
    unadj_se = unadj_model.bse["female"]
    unadj_p = unadj_model.pvalues["female"]
    unadj_or = float(np.exp(unadj_coef))
    unadj_ci = np.exp(unadj_coef + np.array([-1, 1]) * 1.96 * unadj_se)
else:
    unadj_model = smf.logit("deny ~ female", data=df).fit(disp=False)
    unadj_coef = unadj_model.params["female"]
    unadj_se = unadj_model.bse["female"]
    unadj_p = unadj_model.pvalues["female"]
    unadj_or = float(np.exp(unadj_coef))
    unadj_ci = np.exp(unadj_coef + np.array([-1, 1]) * 1.96 * unadj_se)

# Adjusted model controlling for credit-related covariates
control_vars = [
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
]
# Keep only controls that exist
control_vars = [c for c in control_vars if c in df.columns]

formula = outcome + " ~ female"
if control_vars:
    formula += " + " + " + ".join(control_vars)

adj_model = smf.logit(formula, data=df).fit(disp=False)
adj_coef = adj_model.params["female"]
adj_se = adj_model.bse["female"]
adj_p = adj_model.pvalues["female"]
adj_or = float(np.exp(adj_coef))
adj_ci = np.exp(adj_coef + np.array([-1, 1]) * 1.96 * adj_se)

results = {
    "outcome": outcome,
    "rate_by_gender": rate_by_gender.to_dict(),
    "count_by_gender": count_by_gender.to_dict(),
    "chi2_p": float(chi2_p),
    "unadjusted": {
        "odds_ratio": unadj_or,
        "p_value": float(unadj_p),
        "ci_low": float(unadj_ci[0]),
        "ci_high": float(unadj_ci[1]),
    },
    "adjusted": {
        "odds_ratio": adj_or,
        "p_value": float(adj_p),
        "ci_low": float(adj_ci[0]),
        "ci_high": float(adj_ci[1]),
    },
}

print(json.dumps(results, indent=2))
