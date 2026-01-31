import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning: drop rows with missing required fields
required_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = _df.dropna(subset=required_cols).copy()

# Ensure valid counts
# Keep rows where sockets > 0 and num_amtl between 0 and sockets
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask].copy()

# Binary indicator for modern humans
# Note: genus values include "Homo sapiens"
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Prepare endog for binomial GLM (successes, failures)
df["failures"] = df["sockets"] - df["num_amtl"]

# Fit binomial regression controlling for age, sex estimate, and tooth class
# Use categorical tooth_class
formula = "num_amtl + failures ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

# Extract key results for is_human
coef = model.params.get("is_human", float("nan"))
se = model.bse.get("is_human", float("nan"))
pval = model.pvalues.get("is_human", float("nan"))

# Convert coefficient to odds ratio for interpretability
or_human = float(np.exp(coef)) if np.isfinite(coef) else float("nan")

print("Rows used:", len(df))
print("Is_human coefficient (log-odds):", coef)
print("SE:", se)
print("p-value:", pval)
print("Odds ratio (human vs non-human):", or_human)

# Also compute predicted AMTL rate at mean covariates for human vs non-human
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()
# choose most common tooth_class for representative comparison
mode_tooth_class = df["tooth_class"].mode().iloc[0]

pred_df = pd.DataFrame({
    "is_human": [0, 1],
    "age": [mean_age, mean_age],
    "prob_male": [mean_prob_male, mean_prob_male],
    "tooth_class": [mode_tooth_class, mode_tooth_class],
    "num_amtl": [1, 1],  # placeholder; not used for prediction with formula-based model
    "failures": [1, 1],
})

pred = model.predict(pred_df)
print("Predicted AMTL rate at mean covariates (non-human, human):", pred.tolist())
