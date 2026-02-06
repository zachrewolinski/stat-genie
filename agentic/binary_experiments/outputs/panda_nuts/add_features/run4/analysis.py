import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure positive session length
if (df["seconds"] <= 0).any():
    df = df[df["seconds"] > 0].copy()

# Define efficiency as nuts opened per second
# Avoid division by zero already handled

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Standardize age for stability
if df["age"].std() > 0:
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
else:
    df["age_z"] = 0.0

# Model: efficiency ~ age + sex + help
# Use robust standard errors due to potential heteroskedasticity
model = smf.ols("efficiency ~ age_z + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Save model summary and key stats for later inspection
with open("analysis_results.txt", "w") as f:
    f.write(model.summary().as_text())
    f.write("\n\n")
    f.write("Coefficients:\n")
    f.write(model.params.to_string())
    f.write("\n\nP-values:\n")
    f.write(model.pvalues.to_string())

print(model.summary())
