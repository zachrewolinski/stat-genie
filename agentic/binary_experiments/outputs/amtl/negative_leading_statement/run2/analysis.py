import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
# Keep valid rows where sockets and num_amtl make sense
initial_n = len(df)
df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

# Create response as proportion and weights
# Binary indicator for modern humans

df["human"] = (df["genus"] == "Homo sapiens").astype(int)
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Fit binomial GLM with logit link
formula = "amtl_rate ~ human + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
result = model.fit()

# Extract key statistics for the human effect
coef = result.params.get("human", np.nan)
se = result.bse.get("human", np.nan)
pval = result.pvalues.get("human", np.nan)

odds_ratio = np.exp(coef) if np.isfinite(coef) else np.nan

print("Rows used:", len(df), "(from", initial_n, ")")
print(result.summary())
print("\nHuman effect:")
print("  coef:", coef)
print("  se:", se)
print("  p-value:", pval)
print("  odds ratio:", odds_ratio)
