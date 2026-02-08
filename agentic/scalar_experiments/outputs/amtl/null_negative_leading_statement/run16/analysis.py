import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure no zero sockets (shouldn't be) and drop missing key fields
key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=key_cols).copy()

# Human indicator
df["is_human"] = (df["genus"].str.strip().str.lower() == "homo sapiens").astype(int)

# Response as proportion with binomial weights
# Avoid division by zero just in case
valid = df["sockets"] > 0
df = df.loc[valid].copy()
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Fit GLM binomial
formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
result = model.fit()

# Extract effect for is_human
coef = result.params.get("is_human", float("nan"))
se = result.bse.get("is_human", float("nan"))
pval = result.pvalues.get("is_human", float("nan"))

# Convert to odds ratio for context
odds_ratio = float("nan")
if pd.notna(coef):
    odds_ratio = float(np.exp(coef))

# Map to Likert scale [-100, 100]
# Heuristic based on sign, p-value, and effect size
# Negative coef -> humans lower AMTL; Positive -> higher AMTL
score = 0
if pd.notna(coef) and pd.notna(pval):
    # Base magnitude from p-value
    if pval < 1e-6:
        mag = 90
    elif pval < 1e-4:
        mag = 80
    elif pval < 1e-3:
        mag = 70
    elif pval < 1e-2:
        mag = 60
    elif pval < 5e-2:
        mag = 50
    elif pval < 0.1:
        mag = 30
    else:
        mag = 10

    # Adjust magnitude by effect size (odds ratio distance from 1)
    if pd.notna(odds_ratio):
        effect = abs(odds_ratio - 1.0)
        if effect >= 1.0:
            mag = min(100, mag + 20)
        elif effect >= 0.5:
            mag = min(100, mag + 10)
        elif effect < 0.1:
            mag = max(5, mag - 10)

    score = -mag if coef < 0 else mag

# If coef is positive but non-significant, lean slightly positive; if negative non-significant, slightly negative
if pd.notna(coef) and pd.notna(pval) and pval >= 0.1:
    score = -10 if coef < 0 else 10

# Clamp to [-100, 100] and integer
score = int(max(-100, min(100, round(score))))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Print summary for verification
print("coef_is_human", coef)
print("se_is_human", se)
print("pval_is_human", pval)
print("odds_ratio_is_human", odds_ratio)
print("score", score)
