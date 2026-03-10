import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
n_rows = len(df)

# Define variables
# feature2: 1 female, 0 male
# feature14: 1 accepted, 0 denied
# feature11: 1 denied, 0 accepted

# Validate acceptance/denial complementarity
if "feature14" in df.columns and "feature11" in df.columns:
    complement = (df["feature14"] + df["feature11"]).unique()
else:
    complement = None

# Approval rates by gender
approval_by_gender = df.groupby("feature2")["feature14"].mean()
counts_by_gender = df.groupby("feature2")["feature14"].agg(["count", "sum"]).rename(columns={"sum": "approved"})

# Contingency table for chi-square
ct = pd.crosstab(df["feature2"], df["feature14"])
chi2, pval_chi2, dof, expected = stats.chi2_contingency(ct)

# Difference in approval rates (female - male)
rate_female = approval_by_gender.loc[1] if 1 in approval_by_gender.index else np.nan
rate_male = approval_by_gender.loc[0] if 0 in approval_by_gender.index else np.nan
rate_diff = rate_female - rate_male

# Unadjusted odds ratio (female vs male)
# Construct 2x2: rows gender (0 male, 1 female), cols approved (1) vs denied (0)
# Using ct with columns {0,1}
if set(ct.columns) == {0,1} and set(ct.index) == {0,1}:
    # add 0.5 continuity if any cell zero
    a = ct.loc[1,1]  # female approved
    b = ct.loc[1,0]  # female denied
    c = ct.loc[0,1]  # male approved
    d = ct.loc[0,0]  # male denied
    # continuity correction
    if min(a,b,c,d) == 0:
        a,b,c,d = a+0.5, b+0.5, c+0.5, d+0.5
    odds_ratio = (a/b) / (c/d)
else:
    odds_ratio = np.nan

# Logistic regression controlling for covariates
# Exclude feature1 (likely ID) and feature11 (denied complement) and feature14 (outcome)
# Use remaining numeric features
covariate_cols = [c for c in df.columns if c not in ["feature1", "feature11", "feature14"]]
# Ensure gender included first for reporting
if "feature2" not in covariate_cols:
    covariate_cols = ["feature2"] + covariate_cols

X = df[covariate_cols]
X = sm.add_constant(X, has_constant="add")
y = df["feature14"]

# Drop any rows with missing values
model_df = pd.concat([y, X], axis=1).dropna()
y_clean = model_df["feature14"]
X_clean = model_df.drop(columns=["feature14"])

logit_model = sm.Logit(y_clean, X_clean)
logit_res = logit_model.fit(disp=False)

coef_gender = logit_res.params.get("feature2", np.nan)
pval_gender = logit_res.pvalues.get("feature2", np.nan)

# Convert coefficient to odds ratio
odds_ratio_adj = np.exp(coef_gender) if pd.notnull(coef_gender) else np.nan

# Gather results
results = {
    "n_rows": n_rows,
    "approval_rate_female": float(rate_female),
    "approval_rate_male": float(rate_male),
    "approval_rate_diff_female_minus_male": float(rate_diff),
    "chi2_pvalue": float(pval_chi2),
    "chi2_stat": float(chi2),
    "unadjusted_odds_ratio_female_vs_male": float(odds_ratio),
    "logit_gender_coef": float(coef_gender),
    "logit_gender_pvalue": float(pval_gender),
    "logit_gender_odds_ratio_adj": float(odds_ratio_adj),
    "covariates_used": covariate_cols,
    "n_used_logit": int(len(model_df)),
    "feature11_feature14_complement": complement.tolist() if complement is not None else None
}

print(json.dumps(results, indent=2))
