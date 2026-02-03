import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Keep required columns and drop missing
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = _df[cols].copy()

# Basic cleaning
# Ensure sockets positive and num_amtl not exceeding sockets
# Drop rows with missing or invalid values
mask_valid = (
    df["sockets"].notna()
    & df["num_amtl"].notna()
    & df["age"].notna()
    & df["prob_male"].notna()
    & df["tooth_class"].notna()
    & df["genus"].notna()
)

df = df[mask_valid].copy()

# Ensure numeric types
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["sockets"] > 0].copy()

df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# AMTL rate
# Avoid division by zero (already filtered)
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Fit binomial GLM with weights as number of trials
model = smf.glm(
    "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
).fit()

coef = model.params.get("is_human", np.nan)
se = model.bse.get("is_human", np.nan)
pval = model.pvalues.get("is_human", np.nan)

odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Compute a simple marginal comparison at mean covariates for interpretability
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()
# Use the most common tooth_class as baseline for prediction
baseline_tooth = df["tooth_class"].mode().iloc[0]

pred_df = pd.DataFrame(
    {
        "is_human": [0, 1],
        "age": [mean_age, mean_age],
        "prob_male": [mean_prob_male, mean_prob_male],
        "tooth_class": [baseline_tooth, baseline_tooth],
    }
)

pred = model.predict(pred_df)

# Write conclusion
# Decision rule: humans have higher frequencies if coef > 0 and p < 0.05
is_higher = bool((coef > 0) and (pval < 0.05))

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write("Yes\n" if is_higher else "No\n")
    if np.isfinite(coef):
        direction = "higher" if coef > 0 else "lower"
        f.write(
            f"In a binomial GLM controlling for age, sex (prob_male), and tooth class, the Homo sapiens indicator is {direction} (odds ratio ≈ {odds_ratio:.2f}, p ≈ {pval:.3g}). "
        )
        f.write(
            f"At mean covariates, the model predicts AMTL rate ≈ {pred.iloc[1]:.3f} for humans vs {pred.iloc[0]:.3f} for non-humans.\n"
        )
    else:
        f.write("Model did not produce a valid estimate for the Homo sapiens effect.\n")

# Print a short summary for verification
print(model.summary())
print("\nHomo sapiens effect:")
print(f"  coef = {coef:.4f}, SE = {se:.4f}, p = {pval:.4g}, OR = {odds_ratio:.3f}")
print(f"  Predicted AMTL rate (non-human, baseline tooth class) = {pred.iloc[0]:.4f}")
print(f"  Predicted AMTL rate (human, baseline tooth class) = {pred.iloc[1]:.4f}")
