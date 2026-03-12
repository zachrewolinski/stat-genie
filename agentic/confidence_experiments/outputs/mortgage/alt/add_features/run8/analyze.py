import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Columns relevant to mortgage approval
cols = [
    "accept",
    "deny",
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

# Keep only rows with required columns present
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[cols].copy()

# Ensure numeric, coerce errors to NaN
for c in cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# Drop rows with any missing in selected columns
sub = sub.dropna()

# Basic counts
n_total = len(sub)

# Acceptance rate by gender
rate_by_gender = sub.groupby("female")["accept"].mean()
count_by_gender = sub.groupby("female")["accept"].count()

# Two-proportion z-test / chi-square for difference in acceptance rates
contingency = pd.crosstab(sub["female"], sub["accept"])  # rows: female, cols: accept
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

# Logistic regression: accept ~ female + controls
formula = (
    "accept ~ female + black + housing_expense_ratio + self_employed + "
    "married + mortgage_credit + consumer_credit + bad_history + "
    "PI_ratio + loan_to_value + denied_PMI"
)

robust = smf.logit(formula=formula, data=sub).fit(disp=False, cov_type="HC1")

female_coef = robust.params["female"]
female_se = robust.bse["female"]
female_p = robust.pvalues["female"]

# Odds ratio for female
female_or = np.exp(female_coef)

# Average marginal effect of female on acceptance probability
margeff = robust.get_margeff(at="overall", method="dydx")
me_table = margeff.summary_frame()
female_me = me_table.loc["female", "dy/dx"]
# Handle possible column name variations for p-values
if "P>|z|" in me_table.columns:
    female_me_p = me_table.loc["female", "P>|z|"]
elif "Pr(>|z|)" in me_table.columns:
    female_me_p = me_table.loc["female", "Pr(>|z|)"]
elif "P>|t|" in me_table.columns:
    female_me_p = me_table.loc["female", "P>|t|"]
else:
    female_me_p = float("nan")

# Predicted acceptance rates by gender holding other vars at observed values
# This is equivalent to average predicted probability if we set female=0/1
sub_f0 = sub.copy()
sub_f1 = sub.copy()
sub_f0["female"] = 0
sub_f1["female"] = 1
pred_f0 = robust.predict(sub_f0).mean()
pred_f1 = robust.predict(sub_f1).mean()

results = {
    "n_total": int(n_total),
    "accept_rate_female0": float(rate_by_gender.get(0.0, np.nan)),
    "accept_rate_female1": float(rate_by_gender.get(1.0, np.nan)),
    "count_female0": int(count_by_gender.get(0.0, 0)),
    "count_female1": int(count_by_gender.get(1.0, 0)),
    "chi2_p": float(p_chi2),
    "female_coef": float(female_coef),
    "female_se": float(female_se),
    "female_p": float(female_p),
    "female_or": float(female_or),
    "female_me": float(female_me),
    "female_me_p": float(female_me_p),
    "pred_accept_female0": float(pred_f0),
    "pred_accept_female1": float(pred_f1),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
