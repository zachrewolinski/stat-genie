import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic column handling
cols = [c.strip() for c in df.columns]
if cols != list(df.columns):
    df.columns = cols

# Identify outcome
if "accept" in df.columns:
    outcome = "accept"
elif "deny" in df.columns:
    outcome = "deny"
    # we'll convert to acceptance for interpretability
    df["accept"] = 1 - df["deny"]
    outcome = "accept"
else:
    raise ValueError("No accept/deny column found")

# Required key variable
if "female" not in df.columns:
    raise ValueError("No female column found")

# Candidate controls (if present)
control_candidates = [
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
controls = [c for c in control_candidates if c in df.columns]

# Keep only numeric columns for regression
reg_cols = ["accept", "female"] + controls
reg_df = df[reg_cols].copy()

# Drop missing
reg_df = reg_df.dropna()

# Group stats
grp = reg_df.groupby("female")["accept"].agg(["mean", "count"]).rename(index={0: "male", 1: "female"})

# Chi-square test
contingency = pd.crosstab(reg_df["female"], reg_df["accept"])
chi2, p_chi, dof, exp = chi2_contingency(contingency)

# Logistic regression
X = reg_df[["female"] + controls]
X = sm.add_constant(X, has_constant="add")
y = reg_df["accept"]
logit_model = sm.Logit(y, X)
try:
    result = logit_model.fit(disp=False)
except Exception:
    # fallback to regularized if separation, but report
    result = logit_model.fit_regularized(disp=False)

coef = result.params.get("female", np.nan)
se = result.bse.get("female", np.nan) if hasattr(result, "bse") else np.nan
pval = result.pvalues.get("female", np.nan) if hasattr(result, "pvalues") else np.nan
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Marginal effect: predicted acceptance difference female vs male at mean covariates
means = X.drop(columns=["female"]).mean()
X0 = means.copy()
X0["female"] = 0
X1 = means.copy()
X1["female"] = 1
# Ensure order
X0 = X0[X.columns]
X1 = X1[X.columns]
pr0 = float(result.predict(X0)[0])
pr1 = float(result.predict(X1)[0])

out = {
    "n_rows": int(len(reg_df)),
    "approval_rate_overall": float(reg_df["accept"].mean()),
    "approval_rate_by_gender": {
        "male": float(grp.loc["male", "mean"]) if "male" in grp.index else None,
        "female": float(grp.loc["female", "mean"]) if "female" in grp.index else None,
        "male_n": int(grp.loc["male", "count"]) if "male" in grp.index else None,
        "female_n": int(grp.loc["female", "count"]) if "female" in grp.index else None,
    },
    "chi_square": {
        "chi2": float(chi2),
        "p_value": float(p_chi),
        "dof": int(dof),
    },
    "logit_female": {
        "coef": float(coef),
        "se": float(se) if np.isfinite(se) else None,
        "p_value": float(pval) if np.isfinite(pval) else None,
        "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
        "predicted_accept_male": pr0,
        "predicted_accept_female": pr1,
        "predicted_diff": pr1 - pr0,
    },
    "controls_used": controls,
}

print(json.dumps(out, indent=2))
