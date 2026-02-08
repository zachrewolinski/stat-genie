import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
DF = pd.read_csv("amtl.csv")

# Keep relevant genera and columns
relevant_genera = {"Homo sapiens", "Pan", "Pongo", "Papio"}
DF = DF[DF["genus"].isin(relevant_genera)].copy()

# Drop rows with missing values in modeling columns
model_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
DF = DF.dropna(subset=model_cols)

# Create human indicator
DF["is_human"] = (DF["genus"] == "Homo sapiens").astype(int)

# Guard against invalid socket counts or impossible proportions
DF = DF[DF["sockets"] > 0].copy()
DF = DF[DF["num_amtl"] <= DF["sockets"]].copy()

# Fit binomial GLM on two-column success/failure endog
endog = np.column_stack([DF["num_amtl"], DF["sockets"] - DF["num_amtl"]])
exog = patsy.dmatrix(
    "is_human + age + prob_male + C(tooth_class)",
    DF,
    return_type="dataframe",
)
model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

coef = model.params.get("is_human", np.nan)
se = model.bse.get("is_human", np.nan)
pval = model.pvalues.get("is_human", np.nan)

# If something went wrong, default neutral
if not np.isfinite(coef) or not np.isfinite(se) or not np.isfinite(pval):
    scalar = 0
else:
    sign = 1 if coef > 0 else (-1 if coef < 0 else 0)
    # Evidence strength from p-value
    p_strength = max(0.0, min(1.0, (-np.log10(max(pval, 1e-300))) / 3.0))
    # Effect strength from odds ratio magnitude
    log_or = float(coef)
    e_strength = max(0.0, min(1.0, abs(log_or) / 0.7))  # log(2)~0.693
    strength = 0.5 * p_strength + 0.5 * e_strength
    scalar = int(np.round(sign * 100 * strength))

# Write scalar to conclusion.txt
with open("conclusion.txt", "w", encoding="ascii") as f:
    f.write(str(scalar))

# Save a small JSON summary for traceability (not required, but helpful)
summary = {
    "n_rows": int(len(DF)),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "pval_is_human": float(pval),
    "odds_ratio_is_human": float(np.exp(coef)) if np.isfinite(coef) else None,
    "scalar": int(scalar),
}
with open("analysis_summary.json", "w", encoding="ascii") as f:
    json.dump(summary, f, indent=2)
