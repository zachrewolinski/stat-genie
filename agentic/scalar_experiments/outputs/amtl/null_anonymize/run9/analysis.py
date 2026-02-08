import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("amtl.csv")

# Rename for clarity
col_map = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing_teeth",
    "feature4": "observable_sockets",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex_score",
    "feature8": "genus",
    "feature9": "region",
}

df = df.rename(columns=col_map)

# Basic cleaning
# Keep rows with valid counts
mask = (
    df["missing_teeth"].notna()
    & df["observable_sockets"].notna()
    & df["observable_sockets"].gt(0)
)

# Ensure counts are within bounds
mask &= df["missing_teeth"].ge(0) & df["missing_teeth"].le(df["observable_sockets"])

# Keep needed covariates
mask &= df["age"].notna() & df["sex_score"].notna() & df["tooth_class"].notna() & df["genus"].notna()

clean = df.loc[mask].copy()

# Human indicator
clean["is_human"] = (clean["genus"] == "Homo sapiens").astype(int)

# Proportion for inspection
clean["prop_missing"] = clean["missing_teeth"] / clean["observable_sockets"]

# Fit binomial GLM with logit link
# Use missing_teeth / observable_sockets as response with weights (trials)
formula = "prop_missing ~ is_human + age + sex_score + C(tooth_class)"

model = smf.glm(
    formula=formula,
    data=clean,
    family=sm.families.Binomial(),
    freq_weights=clean["observable_sockets"],
).fit()

# Marginal predicted probabilities for human vs nonhuman
# Keep other covariates at observed values (g-computation)
base = clean.copy()

# Predict for human=1 and human=0
base_human = base.copy()
base_human["is_human"] = 1
base_nonhuman = base.copy()
base_nonhuman["is_human"] = 0

pred_human = model.predict(base_human)
pred_nonhuman = model.predict(base_nonhuman)

mean_human = float(np.mean(pred_human))
mean_nonhuman = float(np.mean(pred_nonhuman))

# Compute difference and ratio
mean_diff = mean_human - mean_nonhuman
mean_ratio = mean_human / mean_nonhuman if mean_nonhuman > 0 else np.nan

# Extract coefficient and p-value for is_human
coef = float(model.params.get("is_human", np.nan))
pval = float(model.pvalues.get("is_human", np.nan))

# Simple effect size in odds ratio
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Print summary metrics
print("N total:", len(df))
print("N used:", len(clean))
print("Mean prop missing (raw):", float(clean["prop_missing"].mean()))
print("Human raw mean:", float(clean.loc[clean["is_human"] == 1, "prop_missing"].mean()))
print("Nonhuman raw mean:", float(clean.loc[clean["is_human"] == 0, "prop_missing"].mean()))
print("Model coef is_human:", coef)
print("Model pval is_human:", pval)
print("Odds ratio is_human:", odds_ratio)
print("Marginal mean human:", mean_human)
print("Marginal mean nonhuman:", mean_nonhuman)
print("Marginal diff:", mean_diff)
print("Marginal ratio:", mean_ratio)
