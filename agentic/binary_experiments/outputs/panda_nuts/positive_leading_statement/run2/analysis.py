import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "panda_nuts.csv"
df = pd.read_csv(DATA_PATH)

# Basic cleanup
# Ensure categorical types
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# Create efficiency metric (nuts per second)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Poisson GLM with offset to model rate (nuts per second)
# This directly models nut-cracking efficiency as a rate, adjusted for session length.
formula = "nuts_opened ~ age + C(sex) + C(help)"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])  # rate per second
).fit()

print("Poisson GLM (rate with offset log(seconds))")
print(model.summary())

# Also fit a simple OLS on efficiency for sanity check
ols = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()
print("\nOLS on efficiency (nuts per second)")
print(ols.summary())

# Extract key p-values
pvals = model.pvalues
print("\nPoisson GLM p-values:")
print(pvals)

# Basic conclusion heuristic: any of age, sex, help significant at 0.05?
key_terms = ["age", "C(sex)[T.m]", "C(help)[T.y]"]
found = {k: float(pvals.get(k, np.nan)) for k in key_terms}
print("\nKey term p-values:")
print(found)

sig_any = any((np.isfinite(v) and v < 0.05) for v in found.values())
print("\nAny significant at 0.05?", sig_any)
