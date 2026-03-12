import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Keep relevant columns
cols = [
    "female",
    "accept",
    "deny",
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

# Drop rows with missing values in required columns
_df = _df[cols].dropna().copy()

# Basic counts
n_total = len(_df)

# Approval rate by gender
rate_female = _df.loc[_df["female"] == 1, "accept"].mean()
rate_male = _df.loc[_df["female"] == 0, "accept"].mean()

# Two-proportion z-test (female vs male) on acceptance
success_f = _df.loc[_df["female"] == 1, "accept"].sum()
count_f = (_df["female"] == 1).sum()
success_m = _df.loc[_df["female"] == 0, "accept"].sum()
count_m = (_df["female"] == 0).sum()

# Pooled proportion
p_pool = (success_f + success_m) / (count_f + count_m)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / count_f + 1 / count_m))
if se_pool > 0:
    z_stat = (success_f / count_f - success_m / count_m) / se_pool
    p_two_prop = 2 * (1 - stats.norm.cdf(abs(z_stat)))
else:
    z_stat = np.nan
    p_two_prop = np.nan

# Logistic regression: acceptance ~ female + controls
formula = (
    "accept ~ female + black + housing_expense_ratio + self_employed + married "
    "+ mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value"
)

model = smf.logit(formula=formula, data=_df).fit(disp=False)

# Extract female effect
coef_female = model.params.get("female", np.nan)
se_female = model.bse.get("female", np.nan)
p_female = model.pvalues.get("female", np.nan)

# Odds ratio and 95% CI
or_female = np.exp(coef_female)
ci_low = np.exp(coef_female - 1.96 * se_female) if np.isfinite(se_female) else np.nan
ci_high = np.exp(coef_female + 1.96 * se_female) if np.isfinite(se_female) else np.nan

# Marginal effect at means (approx)
means = _df.drop(columns=["accept", "deny"]).mean()
# Build prediction for male and female at mean covariates
X_base = means.copy()
X_base["female"] = 0.0
X_f = means.copy()
X_f["female"] = 1.0

# Add intercept for prediction
# Use model.predict with a DataFrame
pred_m = model.predict(pd.DataFrame([X_base]))[0]
pred_f = model.predict(pd.DataFrame([X_f]))[0]

marginal_diff = pred_f - pred_m

# Save results to a json for later use (optional)
results = {
    "n_total": int(n_total),
    "rate_female": float(rate_female),
    "rate_male": float(rate_male),
    "two_prop_z": float(z_stat),
    "two_prop_p": float(p_two_prop),
    "coef_female": float(coef_female),
    "se_female": float(se_female),
    "p_female": float(p_female),
    "or_female": float(or_female),
    "or_ci_low": float(ci_low),
    "or_ci_high": float(ci_high),
    "pred_accept_male_mean": float(pred_m),
    "pred_accept_female_mean": float(pred_f),
    "marginal_diff": float(marginal_diff),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
