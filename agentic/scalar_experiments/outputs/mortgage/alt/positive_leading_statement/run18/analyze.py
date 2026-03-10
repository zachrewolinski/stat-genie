import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv("mortgage.csv")

# Basic cleanup
# Ensure binary ints
for col in ["female", "accept", "deny"]:
    if col in df.columns:
        df[col] = df[col].astype(float)

# Create outcome: accept (1=accepted)
if "accept" in df.columns:
    y = df["accept"]
elif "deny" in df.columns:
    y = 1 - df["deny"]
else:
    raise ValueError("No accept or deny column found")

# Unadjusted difference in acceptance rates by gender
female = df["female"]
# counts
accept_f = y[female == 1].sum()
accept_m = y[female == 0].sum()
count_f = (female == 1).sum()
count_m = (female == 0).sum()

# two-proportion z-test
count = np.array([accept_f, accept_m])
obs = np.array([count_f, count_m])
stat, pval = proportions_ztest(count, obs)

rate_f = accept_f / count_f if count_f else np.nan
rate_m = accept_m / count_m if count_m else np.nan
rate_diff = rate_f - rate_m

# Adjusted logistic regression
# Choose covariates that reflect applicant characteristics; exclude 'deny' to avoid leakage
covariates = [
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

available_covariates = [c for c in covariates if c in df.columns]
X = df[available_covariates].copy()
X = sm.add_constant(X, has_constant="add")

# Fit logistic regression with robust SE (HC1)
model = sm.Logit(y, X, missing="drop")
# Fit with robust (HC1) standard errors if supported by this statsmodels version
try:
    robust = model.fit(disp=False, cov_type="HC1")
except TypeError:
    robust = model.fit(disp=False)

# Extract female coefficient
if "female" in robust.params.index:
    coef = robust.params["female"]
    se = robust.bse["female"]
    pval_f = robust.pvalues["female"]
    # Odds ratio and 95% CI
    or_f = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
else:
    coef = se = pval_f = or_f = ci_low = ci_high = np.nan

# Save a small summary for manual review
summary = {
    "n": int(len(df)),
    "female_count": int(count_f),
    "male_count": int(count_m),
    "accept_rate_female": float(rate_f),
    "accept_rate_male": float(rate_m),
    "accept_rate_diff_f_minus_m": float(rate_diff),
    "unadjusted_ztest_pvalue": float(pval),
    "logit_female_coef": float(coef),
    "logit_female_or": float(or_f),
    "logit_female_or_ci": [float(ci_low), float(ci_high)],
    "logit_female_pvalue": float(pval_f),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
