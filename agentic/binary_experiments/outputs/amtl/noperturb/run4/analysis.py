import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning: drop rows with missing essential fields
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
_df = _df.dropna(subset=cols).copy()

# Create binary indicator for modern humans
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Avoid division by zero if any (shouldn't be, given min sockets 2)
_df = _df[_df["sockets"] > 0]

# Binomial GLM with logit link; use proportion with weights (sockets)
_df["prop_amtl"] = _df["num_amtl"] / _df["sockets"]

formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df["sockets"],
).fit()

print(model.summary())

# Extract effect for is_human
coef = model.params["is_human"]
se = model.bse["is_human"]
pval = model.pvalues["is_human"]

# Odds ratio and 95% CI
import numpy as np

or_val = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

print("\nHuman effect (log-odds):", coef)
print("SE:", se)
print("p-value:", pval)
print("Odds ratio:", or_val)
print("95% CI:", (ci_low, ci_high))
