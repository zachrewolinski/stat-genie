import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF_PATH = "amtl.csv"

df = pd.read_csv(DF_PATH)

# Rename columns for clarity
colmap = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing",
    "feature4": "observed",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}

df = df.rename(columns=colmap)

# Basic cleaning: drop rows with missing key fields or invalid counts
# Ensure observed >= missing and observed > 0
mask_valid = (
    df["missing"].notna()
    & df["observed"].notna()
    & df["age"].notna()
    & df["sex"].notna()
    & df["tooth_class"].notna()
    & df["genus"].notna()
)

df = df.loc[mask_valid].copy()

# Ensure numeric
for c in ["missing", "observed", "age", "sex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.loc[df["observed"] > 0].copy()
# enforce missing <= observed
# If any rows violate, drop them

df = df.loc[df["missing"] <= df["observed"]].copy()

# Binary indicator for human
# In data, genus has "Homo sapiens"

df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Prepare endog as successes/failures
# success = missing, failure = observed - missing

df["non_missing"] = df["observed"] - df["missing"]

# Build formula with categorical tooth_class; sex treated as numeric (0-1)
formula = "missing + non_missing ~ is_human + age + sex + C(tooth_class)"

# Fit GLM binomial
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
)

result = model.fit()

# Extract human coefficient
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)
pval = result.pvalues.get("is_human", np.nan)

# Odds ratio
odds_ratio = float(np.exp(coef)) if pd.notna(coef) else np.nan

# Compute predicted probabilities for human vs non-human at mean covariates
# For categorical tooth_class, use most common category

tooth_mode = df["tooth_class"].mode(dropna=True)
if len(tooth_mode) == 0:
    tooth_ref = df["tooth_class"].iloc[0]
else:
    tooth_ref = tooth_mode.iloc[0]

mean_age = df["age"].mean()
mean_sex = df["sex"].mean()

pred_df = pd.DataFrame(
    {
        "is_human": [0, 1],
        "age": [mean_age, mean_age],
        "sex": [mean_sex, mean_sex],
        "tooth_class": [tooth_ref, tooth_ref],
        "missing": [0, 0],
        "non_missing": [1, 1],
    }
)

pred = result.predict(pred_df)
nonhuman_rate = float(pred.iloc[0])
human_rate = float(pred.iloc[1])
rate_diff = human_rate - nonhuman_rate

# Also compute model-based marginal effect of human via delta method (approx)
# We will not overcomplicate; use the sign and significance.

summary = {
    "n_rows": int(df.shape[0]),
    "odds_ratio": odds_ratio,
    "coef": float(coef),
    "se": float(se),
    "pval": float(pval),
    "nonhuman_rate": nonhuman_rate,
    "human_rate": human_rate,
    "rate_diff": rate_diff,
    "tooth_class_ref": tooth_ref,
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
