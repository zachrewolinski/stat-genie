import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv("mortgage.csv")

# Drop unnamed index-like columns
for col in list(_df.columns):
    if col.lower().startswith("unnamed"):
        _df = _df.drop(columns=[col])

# Determine outcome column
if "accept" in _df.columns:
    outcome = "accept"
elif "deny" in _df.columns:
    outcome = "deny"
else:
    raise ValueError("No accept or deny column found")

# Basic cleanup
_df = _df.copy()

# Ensure female is binary 0/1
if "female" not in _df.columns:
    raise ValueError("No female column found")

# Unadjusted group stats
_group = _df.groupby("female")[outcome].agg(["count", "mean"]).rename(columns={"mean": "rate"})

# Align male=0, female=1 if possible
n0 = _group.loc[0, "count"] if 0 in _group.index else np.nan
p0 = _group.loc[0, "rate"] if 0 in _group.index else np.nan
n1 = _group.loc[1, "count"] if 1 in _group.index else np.nan
p1 = _group.loc[1, "rate"] if 1 in _group.index else np.nan

# Proportions z-test for difference in approval rates
if np.isfinite(n0) and np.isfinite(n1):
    successes = np.array([p1 * n1, p0 * n0])
    ns = np.array([n1, n0])
    zstat, pval = proportions_ztest(successes, ns)
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
else:
    zstat = pval = diff = se = ci_low = ci_high = np.nan

# Adjusted logistic regression
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

covariates = [c for c in covariates if c in _df.columns]

reg_df = _df[covariates + [outcome]].dropna()

X = reg_df[covariates]
X = sm.add_constant(X, has_constant="add")

y = reg_df[outcome]

# If outcome is deny, re-interpret to approval probability for consistency
# but keep model on original outcome; report effect on outcome as-is
model = sm.Logit(y, X).fit(disp=False)

coef = model.params.get("female", np.nan)
pval_female = model.pvalues.get("female", np.nan)

# Odds ratio and CI
if np.isfinite(coef):
    conf = model.conf_int().loc["female"]
    or_female = float(np.exp(coef))
    or_ci_low = float(np.exp(conf[0]))
    or_ci_high = float(np.exp(conf[1]))
else:
    or_female = or_ci_low = or_ci_high = np.nan

results = {
    "outcome": outcome,
    "group_stats": {
        "male_count": float(n0),
        "male_rate": float(p0),
        "female_count": float(n1),
        "female_rate": float(p1),
        "diff_female_minus_male": float(diff),
        "diff_ci_low": float(ci_low),
        "diff_ci_high": float(ci_high),
        "z_stat": float(zstat),
        "p_value": float(pval),
    },
    "logit": {
        "n": int(reg_df.shape[0]),
        "coef_female": float(coef),
        "p_value": float(pval_female),
        "odds_ratio": float(or_female),
        "or_ci_low": float(or_ci_low),
        "or_ci_high": float(or_ci_high),
    },
}

print(json.dumps(results, indent=2))
