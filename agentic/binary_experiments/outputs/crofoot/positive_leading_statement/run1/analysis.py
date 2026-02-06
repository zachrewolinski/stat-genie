import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "crofoot.csv"

df = pd.read_csv(DATA_PATH)

# Construct predictors
# Relative group size: focal minus other
# Contest location: relative proximity to home range center
# Positive location_diff means the contest is closer to focal than to other

df = df.copy()

df["size_diff"] = df["n_focal"] - df["n_other"]
df["location_diff"] = df["dist_other"] - df["dist_focal"]

# Fit logistic regression
model = smf.glm(
    formula="win ~ size_diff + location_diff",
    data=df,
    family=sm.families.Binomial(),
).fit()

# Print summary for inspection
print(model.summary())

# Also compute simple descriptive stats by size and location difference
print("\nDescriptive stats:")
print(df[["win", "size_diff", "location_diff"]].describe())

# Compute marginal effect sign and p-values for quick decision
params = model.params
pvalues = model.pvalues

print("\nCoefficients:")
print(params)
print("\nP-values:")
print(pvalues)

# Save key results for downstream use
results = pd.DataFrame({
    "coef": params,
    "pvalue": pvalues,
})
results.to_csv("analysis_results.csv")
