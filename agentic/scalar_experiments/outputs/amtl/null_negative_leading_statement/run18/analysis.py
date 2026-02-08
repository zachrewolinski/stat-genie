import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning: drop rows with missing essential fields
needed = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = _df.dropna(subset=needed).copy()

# Create human indicator
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Guard against invalid sockets
valid = df["sockets"] > 0
if not valid.all():
    df = df.loc[valid].copy()

# Proportion response for binomial GLM
df["amtl_prop"] = df["num_amtl"] / df["sockets"]

# Encode tooth_class as categorical
# Use treatment coding with 'Posterior' as baseline if present
if "Posterior" in df["tooth_class"].unique():
    df["tooth_class"] = pd.Categorical(df["tooth_class"], categories=["Posterior", "Anterior", "Premolar"], ordered=False)

# Fit binomial GLM with frequency weights
model = smf.glm(
    formula="amtl_prop ~ is_human + age + prob_male + tooth_class",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
res = model.fit()

# Extract human effect
coef = res.params.get("is_human", np.nan)
se = res.bse.get("is_human", np.nan)
pval = res.pvalues.get("is_human", np.nan)

# Odds ratio
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Compute average marginal effect by toggling human indicator
# Predict at observed covariates
pred_nonhuman = res.predict(df.assign(is_human=0))
pred_human = res.predict(df.assign(is_human=1))

avg_diff = float(np.mean(pred_human - pred_nonhuman))

# Save summary stats for downstream use
summary = {
    "coef": float(coef),
    "se": float(se),
    "pval": float(pval),
    "odds_ratio": float(odds_ratio),
    "avg_diff": avg_diff,
    "n": int(df.shape[0]),
}

print("SUMMARY", summary)
