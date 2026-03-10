import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleanup: drop unnamed index if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure binary columns are numeric
binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing key fields
key_cols = ["female", "accept"]
analysis_df = df.dropna(subset=key_cols)

# Descriptive stats: acceptance rates by gender
rate_by_gender = (
    analysis_df.groupby("female")["accept"]
    .agg(["mean", "count"])
    .rename(index={0: "male", 1: "female"})
)

# Two-proportion z-test (female vs male acceptance)
# contingency table
contingency = pd.crosstab(analysis_df["female"], analysis_df["accept"])
# ensure both outcomes present
if contingency.shape == (2, 2):
    # counts
    female_accept = contingency.loc[1, 1]
    female_total = contingency.loc[1].sum()
    male_accept = contingency.loc[0, 1]
    male_total = contingency.loc[0].sum()
    # proportions
    p1 = female_accept / female_total
    p2 = male_accept / male_total
    # pooled proportion
    p_pool = (female_accept + male_accept) / (female_total + male_total)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / female_total + 1 / male_total))
    z = (p1 - p2) / se if se > 0 else np.nan
    p_value_z = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
else:
    p1 = p2 = z = p_value_z = np.nan

# Logistic regression: unadjusted and adjusted
# Unadjusted
X_unadj = sm.add_constant(analysis_df[["female"]])
model_unadj = sm.Logit(analysis_df["accept"], X_unadj, missing="drop")
res_unadj = model_unadj.fit(disp=False)

# Adjusted model with available covariates (exclude deny to avoid redundancy with accept)
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
# Keep only columns present
covariates = [c for c in covariates if c in analysis_df.columns]

adj_df = analysis_df.dropna(subset=covariates + ["accept"])
X_adj = sm.add_constant(adj_df[covariates])
model_adj = sm.Logit(adj_df["accept"], X_adj, missing="drop")
res_adj = model_adj.fit(disp=False)

# Extract female coefficient stats
unadj_coef = res_unadj.params.get("female", np.nan)
unadj_se = res_unadj.bse.get("female", np.nan)
unadj_p = res_unadj.pvalues.get("female", np.nan)

adj_coef = res_adj.params.get("female", np.nan)
adj_se = res_adj.bse.get("female", np.nan)
adj_p = res_adj.pvalues.get("female", np.nan)

# Odds ratios
unadj_or = float(np.exp(unadj_coef)) if np.isfinite(unadj_coef) else np.nan
adj_or = float(np.exp(adj_coef)) if np.isfinite(adj_coef) else np.nan

# Assemble results
results = {
    "n_total": int(len(analysis_df)),
    "rate_by_gender": rate_by_gender.to_dict(),
    "diff_in_acceptance": float(p1 - p2) if np.isfinite(p1) and np.isfinite(p2) else np.nan,
    "z_test": {
        "z": float(z) if np.isfinite(z) else np.nan,
        "p_value": float(p_value_z) if np.isfinite(p_value_z) else np.nan,
    },
    "logit_unadjusted": {
        "coef_female": float(unadj_coef) if np.isfinite(unadj_coef) else np.nan,
        "se_female": float(unadj_se) if np.isfinite(unadj_se) else np.nan,
        "p_value_female": float(unadj_p) if np.isfinite(unadj_p) else np.nan,
        "odds_ratio_female": unadj_or,
    },
    "logit_adjusted": {
        "n": int(len(adj_df)),
        "coef_female": float(adj_coef) if np.isfinite(adj_coef) else np.nan,
        "se_female": float(adj_se) if np.isfinite(adj_se) else np.nan,
        "p_value_female": float(adj_p) if np.isfinite(adj_p) else np.nan,
        "odds_ratio_female": adj_or,
    },
}

print(json.dumps(results, indent=2))
