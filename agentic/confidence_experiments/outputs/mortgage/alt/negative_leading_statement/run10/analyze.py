import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleanup: ensure binary columns are numeric 0/1
binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing in key columns
key_cols = ["female", "accept"]
analysis_df = df.dropna(subset=key_cols).copy()

# 1) Unadjusted association: approval rate by gender
ct = pd.crosstab(analysis_df["female"], analysis_df["accept"], dropna=False)
# Ensure columns order 0,1
ct = ct.reindex(index=[0,1], columns=[0,1], fill_value=0)

# Approval rates
female0_total = ct.loc[0].sum()
female1_total = ct.loc[1].sum()
rate_male = ct.loc[0,1] / female0_total if female0_total > 0 else np.nan
rate_female = ct.loc[1,1] / female1_total if female1_total > 0 else np.nan
rate_diff = rate_female - rate_male

# Two-proportion z-test / chi-square
# Use chi-square test of independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct.values)

# 2) Logistic regression: accept ~ female + controls
# Controls: all other applicant/lender characteristics provided
control_cols = [
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

# Keep only columns that exist and are non-null
model_cols = ["accept", "female"] + [c for c in control_cols if c in analysis_df.columns]
model_df = analysis_df[model_cols].dropna().copy()

X = model_df.drop(columns=["accept"])
X = sm.add_constant(X, has_constant="add")
y = model_df["accept"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Extract female coefficient, odds ratio and p-value
female_coef = result.params.get("female", np.nan)
female_se = result.bse.get("female", np.nan)
female_p = result.pvalues.get("female", np.nan)

odds_ratio = np.exp(female_coef) if pd.notnull(female_coef) else np.nan

# Average marginal effect via predicted probabilities
try:
    X0 = X.copy()
    X1 = X.copy()
    if "female" in X0.columns:
        X0["female"] = 0
        X1["female"] = 1
    p0 = result.predict(X0)
    p1 = result.predict(X1)
    female_marg = float(np.mean(p1 - p0))
    female_marg_p = np.nan
except Exception:
    female_marg = np.nan
    female_marg_p = np.nan

output = {
    "n_total": int(len(analysis_df)),
    "n_model": int(len(model_df)),
    "approval_rate_male": rate_male,
    "approval_rate_female": rate_female,
    "approval_rate_diff_female_minus_male": rate_diff,
    "chi2_p_value": p_chi2,
    "logit_female_coef": female_coef,
    "logit_female_se": female_se,
    "logit_female_p": female_p,
    "logit_female_odds_ratio": odds_ratio,
    "logit_female_marginal_effect": female_marg,
    "logit_female_marginal_p": female_marg_p,
}

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
