import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop obvious index-like columns
for col in ["Unnamed: 0", "unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Ensure accept exists
if "accept" not in df.columns and "deny" in df.columns:
    df["accept"] = 1 - df["deny"]

# Basic sanity checks
required = ["female", "accept"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# Keep only rows with non-missing female/accept
df = df.copy()
base = df[["female", "accept"]].dropna()

# Approval rates by gender
rate_by_gender = base.groupby("female")["accept"].mean()
count_by_gender = base.groupby("female")["accept"].count()

# Chi-square test
cont_table = pd.crosstab(base["female"], base["accept"])
chi2, p_chi2, dof, expected = chi2_contingency(cont_table)

# Logistic regression: unadjusted
X_unadj = sm.add_constant(base[["female"]])
model_unadj = sm.Logit(base["accept"], X_unadj, missing="drop").fit(disp=False)

# Logistic regression: adjusted with available covariates
core_covs = [
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

extra_covs = ["age", "occupation"]

covs_core = [c for c in core_covs if c in df.columns]
covs_full = covs_core + [c for c in extra_covs if c in df.columns]

model_adj = None
margeff = None
avg_pred_diff = None
model_adj_full = None
avg_pred_diff_full = None

if covs_core:
    model_df = df[["accept", "female"] + covs_core].dropna()
    X_adj = sm.add_constant(model_df[["female"] + covs_core])
    model_adj = sm.Logit(model_df["accept"], X_adj, missing="drop").fit(disp=False)
    try:
        margeff = model_adj.get_margeff(at="overall", method="dydx")
    except Exception:
        margeff = None
    # Average predicted approval difference when flipping female 0->1
    try:
        X1 = X_adj.copy()
        X0 = X_adj.copy()
        X1["female"] = 1
        X0["female"] = 0
        pred1 = model_adj.predict(X1)
        pred0 = model_adj.predict(X0)
        avg_pred_diff = float((pred1 - pred0).mean())
    except Exception:
        avg_pred_diff = None

if covs_full and covs_full != covs_core:
    model_df_full = df[["accept", "female"] + covs_full].dropna()
    X_full = sm.add_constant(model_df_full[["female"] + covs_full])
    model_adj_full = sm.Logit(model_df_full["accept"], X_full, missing="drop").fit(disp=False)
    try:
        X1 = X_full.copy()
        X0 = X_full.copy()
        X1["female"] = 1
        X0["female"] = 0
        pred1 = model_adj_full.predict(X1)
        pred0 = model_adj_full.predict(X0)
        avg_pred_diff_full = float((pred1 - pred0).mean())
    except Exception:
        avg_pred_diff_full = None

results = {
    "n_total": int(df.shape[0]),
    "n_used_base": int(base.shape[0]),
    "approval_rate_male": float(rate_by_gender.get(0, np.nan)),
    "approval_rate_female": float(rate_by_gender.get(1, np.nan)),
    "count_male": int(count_by_gender.get(0, 0)),
    "count_female": int(count_by_gender.get(1, 0)),
    "chi2_p_value": float(p_chi2),
    "unadj_female_coef": float(model_unadj.params["female"]),
    "unadj_female_p": float(model_unadj.pvalues["female"]),
}

if model_adj is not None:
    results.update({
        "adj_female_coef": float(model_adj.params["female"]),
        "adj_female_p": float(model_adj.pvalues["female"]),
        "adj_n": int(model_adj.nobs),
        "adj_covariates": covs_core,
    })
    if avg_pred_diff is not None:
        results["adj_avg_pred_diff_female_minus_male"] = avg_pred_diff
    if margeff is not None:
        try:
            mfx = margeff.summary_frame()
            if "female" in mfx.index:
                results.update({
                    "adj_female_marginal_effect": float(mfx.loc["female", "dy/dx"]),
                    "adj_female_marginal_p": float(mfx.loc["female", "P>|z|"]),
                })
        except Exception:
            pass

if model_adj_full is not None:
    results.update({
        "full_adj_female_coef": float(model_adj_full.params["female"]),
        "full_adj_female_p": float(model_adj_full.pvalues["female"]),
        "full_adj_n": int(model_adj_full.nobs),
        "full_adj_covariates": covs_full,
    })
    if avg_pred_diff_full is not None:
        results["full_adj_avg_pred_diff_female_minus_male"] = avg_pred_diff_full

print(json.dumps(results, indent=2))
