import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Map columns
# feature2: female (1) vs male (0)
# feature14: accepted (1) vs denied (0)
# feature11: denied (1) vs accepted (0)

# Basic checks
n_rows = len(df)
missing = df.isna().sum().sum()

# Outcome
if "feature14" in df.columns:
    outcome = df["feature14"]
    outcome_name = "accepted"
else:
    outcome = 1 - df["feature11"]
    outcome_name = "accepted"

# Gender
gender = df["feature2"]

# Drop rows with missing in relevant columns
base_cols = ["feature2", "feature14"] if "feature14" in df.columns else ["feature2", "feature11"]
base = df[base_cols].dropna()

# Approval rates by gender
rate_by_gender = base.groupby("feature2")[base_cols[-1]].mean()
# if outcome is accepted (feature14), then higher is approval
# if using feature11 (denied), then approval = 1 - denied
if base_cols[-1] == "feature11":
    rate_by_gender = 1 - rate_by_gender

# Counts
counts = base.groupby("feature2")[base_cols[-1]].count()

# Two-proportion z-test (female vs male) on approval
# Build success counts
if base_cols[-1] == "feature11":
    successes = base.groupby("feature2")["feature11"].apply(lambda s: (1 - s).sum())
    totals = base.groupby("feature2")["feature11"].size()
else:
    successes = base.groupby("feature2")["feature14"].sum()
    totals = base.groupby("feature2")["feature14"].size()

# Ensure order: male (0), female (1)
success_male = float(successes.get(0, np.nan))
success_female = float(successes.get(1, np.nan))
total_male = float(totals.get(0, np.nan))
total_female = float(totals.get(1, np.nan))

# Proportion z-test
p_pool = (success_male + success_female) / (total_male + total_female)
se = np.sqrt(p_pool * (1 - p_pool) * (1/total_male + 1/total_female))
if se > 0:
    z = (success_female/total_female - success_male/total_male) / se
    p_value_prop = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_value_prop = np.nan

# Logistic regression: acceptance ~ female + controls
# Controls: all features except outcomes, gender, and apparent row identifier
predictors = [c for c in df.columns if c not in {"feature1", "feature2", "feature14", "feature11"}]
reg_cols = ["feature2"] + predictors
reg_df = df[reg_cols + (["feature14"] if "feature14" in df.columns else ["feature11"])].dropna().copy()

# Define y
if "feature14" in reg_df.columns:
    y = reg_df["feature14"]
else:
    y = 1 - reg_df["feature11"]

X = reg_df[reg_cols]
X = sm.add_constant(X, has_constant="add")

# Fit logistic regression
logit_model = sm.Logit(y, X)
try:
    logit_res = logit_model.fit(disp=False)
    coef = logit_res.params["feature2"]
    se_coef = logit_res.bse["feature2"]
    p_value = logit_res.pvalues["feature2"]
    # Odds ratio and 95% CI
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se_coef))
    ci_high = float(np.exp(coef + 1.96 * se_coef))
    llr_pvalue = logit_res.llr_pvalue
    converged = True
except Exception as e:
    coef = np.nan
    se_coef = np.nan
    p_value = np.nan
    odds_ratio = np.nan
    ci_low = np.nan
    ci_high = np.nan
    llr_pvalue = np.nan
    converged = False

results = {
    "n_rows": int(n_rows),
    "missing_values": int(missing),
    "approval_rate_male": float(rate_by_gender.get(0, np.nan)),
    "approval_rate_female": float(rate_by_gender.get(1, np.nan)),
    "count_male": int(counts.get(0, 0)),
    "count_female": int(counts.get(1, 0)),
    "prop_test_z": float(z),
    "prop_test_p": float(p_value_prop),
    "logit_converged": bool(converged),
    "logit_coef_female": float(coef),
    "logit_or_female": float(odds_ratio),
    "logit_or_ci_low": float(ci_low),
    "logit_or_ci_high": float(ci_high),
    "logit_p_female": float(p_value),
    "logit_llr_p": float(llr_pvalue)
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
