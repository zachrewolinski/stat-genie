import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic checks
n_rows = df.shape[0]

# Define outcome: accept (1 if accepted, 0 if denied)
# Ensure binary
outcome = "accept"

# Contingency table for female vs accept
ct = pd.crosstab(df["female"], df[outcome])
chi2, p, dof, expected = stats.chi2_contingency(ct)

# Acceptance rates by gender
rate_by_gender = df.groupby("female")[outcome].mean().to_dict()

# Logistic regression: accept ~ female + controls
# Controls: black, housing_expense_ratio, self_employed, married, mortgage_credit,
# consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI
controls = [
    "female",
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

# Drop missing rows for regression
reg_df = df[controls + [outcome]].dropna()

X = reg_df[controls]
X = sm.add_constant(X, has_constant="add")
y = reg_df[outcome]

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit(disp=0)

coef = logit_result.params
pvalues = logit_result.pvalues

female_coef = coef.get("female", np.nan)
female_p = pvalues.get("female", np.nan)

# Compute odds ratio for female
female_or = float(np.exp(female_coef)) if pd.notnull(female_coef) else np.nan

# Compute marginal effect at means (optional)
try:
    margeff = logit_result.get_margeff(at="mean")
    margeff_summary = margeff.summary_frame()
    female_me = float(margeff_summary.loc["female", "dy/dx"])
    female_me_p = float(margeff_summary.loc["female", "P>|z|"])
except Exception:
    female_me = np.nan
    female_me_p = np.nan

# Output results as JSON to stdout for easy parsing
out = {
    "n_rows": int(n_rows),
    "contingency_table": ct.to_dict(),
    "chi2": float(chi2),
    "chi2_p": float(p),
    "accept_rate_by_female": {str(k): float(v) for k, v in rate_by_gender.items()},
    "logit_female_coef": float(female_coef),
    "logit_female_p": float(female_p),
    "logit_female_or": float(female_or),
    "marginal_effect_female": float(female_me) if pd.notnull(female_me) else None,
    "marginal_effect_female_p": float(female_me_p) if pd.notnull(female_me_p) else None,
    "n_reg_rows": int(reg_df.shape[0]),
    "pseudo_r2": float(logit_result.prsquared),
}

print(json.dumps(out, indent=2))
