import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"

# Read CSV
_df = pd.read_csv(path)

# Normalize column names to lower for consistency
_df.columns = [c.strip() for c in _df.columns]

# Ensure expected columns
required = ["affairs", "children"]
missing = [c for c in required if c not in _df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean children to binary
children_col = _df["children"].astype(str).str.strip().str.lower()
_df["children_yes"] = children_col.isin(["yes", "1", "true", "y"]).astype(int)

# Basic stats
mean_affairs_yes = _df.loc[_df["children_yes"] == 1, "affairs"].mean()
mean_affairs_no = _df.loc[_df["children_yes"] == 0, "affairs"].mean()
mean_diff = mean_affairs_yes - mean_affairs_no

# Any affairs
_df["any_affairs"] = (_df["affairs"] > 0).astype(int)
any_rate_yes = _df.loc[_df["children_yes"] == 1, "any_affairs"].mean()
any_rate_no = _df.loc[_df["children_yes"] == 0, "any_affairs"].mean()
rate_diff = any_rate_yes - any_rate_no

# OLS with controls (if available)
controls = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender"]
controls_present = [c for c in controls if c in _df.columns]

# Prepare data for formulas
# For gender, treat as categorical if present and not numeric
if "gender" in controls_present:
    _df["gender"] = _df["gender"].astype(str)

formula_terms = ["children_yes"]
formula_terms += controls_present
formula = "affairs ~ " + " + ".join(["C(gender)" if c == "gender" else c for c in formula_terms])

ols_result = smf.ols(formula, data=_df).fit(cov_type="HC3")
ols_coef = ols_result.params.get("children_yes", np.nan)
ols_p = ols_result.pvalues.get("children_yes", np.nan)

# Logistic regression for any affairs
logit_formula = "any_affairs ~ " + " + ".join(["C(gender)" if c == "gender" else c for c in formula_terms])
logit_result = smf.logit(logit_formula, data=_df).fit(disp=False)
logit_coef = logit_result.params.get("children_yes", np.nan)
logit_p = logit_result.pvalues.get("children_yes", np.nan)

# Compute effect size (standardized mean difference)
std_affairs = _df["affairs"].std(ddof=0)
std_diff = mean_diff / std_affairs if std_affairs and not np.isnan(std_affairs) else np.nan

summary = {
    "n": len(_df),
    "mean_affairs_yes": mean_affairs_yes,
    "mean_affairs_no": mean_affairs_no,
    "mean_diff_yes_minus_no": mean_diff,
    "any_rate_yes": any_rate_yes,
    "any_rate_no": any_rate_no,
    "any_rate_diff_yes_minus_no": rate_diff,
    "std_affairs": std_affairs,
    "std_diff": std_diff,
    "ols_coef_children_yes": ols_coef,
    "ols_p_children_yes": ols_p,
    "logit_coef_children_yes": logit_coef,
    "logit_p_children_yes": logit_p,
}

for k, v in summary.items():
    print(f"{k}: {v}")
