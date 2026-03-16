import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic integrity: use accept as outcome (1 accepted, 0 denied)
# If accept column is missing or not binary, fall back to (1 - deny)
if "accept" in df.columns:
    y = df["accept"]
else:
    y = 1 - df["deny"]

# Ensure binary numeric
# Drop rows with missing outcome or female
analysis_df = df.copy()
analysis_df["accept_outcome"] = y
analysis_df = analysis_df.dropna(subset=["accept_outcome", "female"])

# Acceptance rates by gender
rate_by_gender = analysis_df.groupby("female")["accept_outcome"].mean().to_dict()
count_by_gender = analysis_df.groupby("female")["accept_outcome"].count().to_dict()

# Contingency table for chi-square: rows female (0/1), cols accept (0/1)
ct = pd.crosstab(analysis_df["female"], analysis_df["accept_outcome"])
chi2, p_chi2, dof, exp = stats.chi2_contingency(ct)

# Unadjusted logistic regression: accept ~ female
X_unadj = sm.add_constant(analysis_df[["female"]])
logit_unadj = sm.Logit(analysis_df["accept_outcome"], X_unadj, missing="drop")
res_unadj = logit_unadj.fit(disp=False)

# Adjusted model with covariates (exclude outcome and index)
# Use available columns; avoid duplication
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

covariates = [c for c in covariates if c in analysis_df.columns]
X_adj = sm.add_constant(analysis_df[covariates])
logit_adj = sm.Logit(analysis_df["accept_outcome"], X_adj, missing="drop")
res_adj = logit_adj.fit(disp=False)

# Extract female effect
female_coef_unadj = res_unadj.params.get("female", np.nan)
female_p_unadj = res_unadj.pvalues.get("female", np.nan)

tmp = res_adj.params.get("female", np.nan)
female_coef_adj = tmp
female_p_adj = res_adj.pvalues.get("female", np.nan)

# Odds ratios
odds_unadj = float(np.exp(female_coef_unadj)) if pd.notna(female_coef_unadj) else np.nan
odds_adj = float(np.exp(female_coef_adj)) if pd.notna(female_coef_adj) else np.nan

# Prepare results
results = {
    "n_rows": int(len(analysis_df)),
    "accept_rate_by_female": {"male_0": rate_by_gender.get(0.0, np.nan), "female_1": rate_by_gender.get(1.0, np.nan)},
    "count_by_female": {"male_0": int(count_by_gender.get(0.0, 0)), "female_1": int(count_by_gender.get(1.0, 0))},
    "chi2_p_value": float(p_chi2),
    "unadjusted": {
        "female_coef": float(female_coef_unadj),
        "female_odds_ratio": float(odds_unadj),
        "female_p_value": float(female_p_unadj),
    },
    "adjusted": {
        "female_coef": float(female_coef_adj),
        "female_odds_ratio": float(odds_adj),
        "female_p_value": float(female_p_adj),
    },
}

print(json.dumps(results, indent=2))
