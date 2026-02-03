import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Create binary human indicator
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Response as proportion with binomial weights
_df["amtl_rate"] = _df["num_amtl"] / _df["sockets"]

# Fit binomial GLM controlling for age, sex (prob_male), and tooth class
formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df["sockets"],
).fit()

# Extract effect for human indicator
coef = model.params["is_human"]
se = model.bse["is_human"]
pval = model.pvalues["is_human"]

# Odds ratio for human vs non-human
odds_ratio = float(np.exp(coef))

print(model.summary())
print("\nHuman indicator coefficient:", coef)
print("Standard error:", se)
print("P-value:", pval)
print("Odds ratio (Homo sapiens vs non-human):", odds_ratio)
