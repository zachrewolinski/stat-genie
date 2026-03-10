import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Map columns based on info.json metadata
GENDER = "feature2"  # 1 female, 0 male
ACCEPT = "feature14"  # 1 accepted, 0 denied
DENIED = "feature11"  # 1 denied, 0 accepted

# Basic sanity: if ACCEPT not binary, fall back to 1-DENIED if present
if ACCEPT in _df.columns:
    y = _df[ACCEPT]
elif DENIED in _df.columns:
    y = 1 - _df[DENIED]
else:
    raise ValueError("No acceptance/denial outcome column found.")

# Ensure binary 0/1
# If values are floats, round to nearest 0/1 for safety if they are close
if not set(np.unique(y.dropna().round()).tolist()).issubset({0, 1}):
    raise ValueError("Outcome is not binary after rounding.")

_df = _df.copy()
_df["accept"] = pd.to_numeric(y, errors="coerce").round()

# Gender series
if GENDER not in _df.columns:
    raise ValueError("Gender column not found.")
_df["female"] = pd.to_numeric(_df[GENDER], errors="coerce").round()

# Drop rows with missing values in outcome or gender
base = _df.dropna(subset=["accept", "female"]).copy()
base["accept"] = base["accept"].astype(int)
base["female"] = base["female"].astype(int)

# Contingency table
ct = pd.crosstab(base["female"], base["accept"])
# Ensure both columns exist
for col in [0, 1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0, 1]]

# Chi-square test
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Approval rates
rate_female = base.loc[base["female"] == 1, "accept"].mean()
rate_male = base.loc[base["female"] == 0, "accept"].mean()
rate_diff = rate_female - rate_male

# Difference in proportions CI (Wald)
# Avoid zero division
n_f = base.loc[base["female"] == 1, "accept"].count()
n_m = base.loc[base["female"] == 0, "accept"].count()
se_diff = np.sqrt(rate_female * (1 - rate_female) / n_f + rate_male * (1 - rate_male) / n_m)
ci_low = rate_diff - 1.96 * se_diff
ci_high = rate_diff + 1.96 * se_diff

# Logistic regression (unadjusted)
X_unadj = sm.add_constant(base[["female"]])
model_unadj = sm.Logit(base["accept"], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Logistic regression (adjusted) using all other features except outcome columns
exclude = {"accept", GENDER, ACCEPT, DENIED}
# Use numeric predictors from dataset
predictors = [c for c in _df.columns if c not in exclude]
# Remove any non-numeric columns just in case
predictors = [c for c in predictors if pd.api.types.is_numeric_dtype(_df[c])]
base_adj = base.dropna(subset=["female"] + predictors).copy()
X_adj = base_adj[["female"] + predictors].copy()
X_adj = sm.add_constant(X_adj)
try:
    model_adj = sm.GLM(base_adj["accept"], X_adj, family=sm.families.Binomial())
    res_adj = model_adj.fit(maxiter=200, disp=False)
except Exception:
    # Fallback to regularized logit if GLM fails to converge
    model_adj = sm.Logit(base_adj["accept"], X_adj)
    res_adj = model_adj.fit_regularized(disp=False)

# Extract effects
coef_unadj = res_unadj.params["female"]
se_unadj = res_unadj.bse["female"]
p_unadj = res_unadj.pvalues["female"]

def _get_named_stat(obj, name):
    try:
        return float(obj[name])
    except Exception:
        try:
            idx = list(res_adj.params.index).index(name)
            return float(obj[idx])
        except Exception:
            return float("nan")

coef_adj = _get_named_stat(res_adj.params, "female")
se_adj = _get_named_stat(res_adj.bse, "female") if hasattr(res_adj, "bse") else float("nan")
p_adj = _get_named_stat(res_adj.pvalues, "female") if hasattr(res_adj, "pvalues") else float("nan")

or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# Build explanation
explanation = {
    "counts": {
        "n_total": int(base.shape[0]),
        "n_female": int(n_f),
        "n_male": int(n_m),
    },
    "approval_rates": {
        "female": float(rate_female),
        "male": float(rate_male),
        "difference_female_minus_male": float(rate_diff),
        "diff_95ci": [float(ci_low), float(ci_high)],
    },
    "chi_square": {
        "chi2": float(chi2),
        "p_value": float(p_chi),
    },
    "logit_unadjusted": {
        "coef": float(coef_unadj),
        "se": float(se_unadj),
        "p_value": float(p_unadj),
        "odds_ratio": float(or_unadj),
    },
    "logit_adjusted": {
        "coef": float(coef_adj),
        "se": float(se_adj),
        "p_value": float(p_adj),
        "odds_ratio": float(or_adj),
    },
}

# Decide Likert response
# Heuristic: rely primarily on adjusted model
response = 50
alpha = 0.05
abs_diff = abs(rate_diff)

# Use adjusted p-value if available; otherwise fall back to unadjusted evidence
p_ref = p_adj if not np.isnan(p_adj) else p_unadj

if p_ref < alpha:
    # Significant; move toward yes based on effect size
    if abs_diff >= 0.10 or abs(np.log(or_adj)) >= np.log(1.5):
        response = 85
    elif abs_diff >= 0.05 or abs(np.log(or_adj)) >= np.log(1.2):
        response = 70
    else:
        response = 60
else:
    # Not significant; move toward no based on how small effect is
    if abs_diff <= 0.02 and 0.9 <= or_adj <= 1.1:
        response = 20
    elif abs_diff <= 0.05 and 0.8 <= or_adj <= 1.25:
        response = 35
    else:
        response = 45

# Compose narrative explanation
adj_note = ""
if np.isnan(p_adj):
    adj_note = "Adjusted model used regularization; p-value not available. "

text = (
    f"Question: Does gender affect mortgage approval?\n"
    f"Sample size: {explanation['counts']['n_total']} applicants; female={explanation['counts']['n_female']}, male={explanation['counts']['n_male']}.\n"
    f"Approval rates: female={explanation['approval_rates']['female']:.3f}, male={explanation['approval_rates']['male']:.3f}, "
    f"difference (female-male)={explanation['approval_rates']['difference_female_minus_male']:.3f} "
    f"with 95% CI [{explanation['approval_rates']['diff_95ci'][0]:.3f}, {explanation['approval_rates']['diff_95ci'][1]:.3f}].\n"
    f"Chi-square test of independence: chi2={explanation['chi_square']['chi2']:.3f}, p={explanation['chi_square']['p_value']:.4f}.\n"
    f"Unadjusted logit (accept ~ female): OR={explanation['logit_unadjusted']['odds_ratio']:.3f}, p={explanation['logit_unadjusted']['p_value']:.4f}.\n"
    f"Adjusted logit (accept ~ female + covariates): OR={explanation['logit_adjusted']['odds_ratio']:.3f}, p={explanation['logit_adjusted']['p_value']:.4f}.\n"
    f"{adj_note}"
    "Interpretation: The adjusted model and approval-rate difference indicate whether gender independently predicts approval "
    "after accounting for creditworthiness and application characteristics. The Likert response reflects the strength and "
    "statistical significance of this evidence."
)

output = {
    "response": int(response),
    "explanation": text,
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)
