import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"

df = pd.read_csv(csv_path)

# Compute efficiency: nuts opened per second
# Avoid division by zero if any duration is zero
if (df["feature6"] <= 0).any():
    raise ValueError("Non-positive session duration found; cannot compute efficiency.")

df = df.copy()

df["efficiency"] = df["feature5"] / df["feature6"]

# Clean categorical labels
# feature3: sex (f/m), feature7: help (y/N)

df["feature3"] = df["feature3"].astype(str)

df["feature7"] = df["feature7"].astype(str)

# Fit OLS model
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit(cov_type="HC3")

# Collect results
params = model.params
pvalues = model.pvalues

# Prepare a concise result table
results = pd.DataFrame({
    "coef": params,
    "p_value": pvalues
})

print("Model: efficiency ~ age + sex + help (OLS, robust HC3)\n")
print(results)
print("\nR-squared:", model.rsquared)

# Also compute group means for interpretability
sex_means = df.groupby("feature3")["efficiency"].mean()
help_means = df.groupby("feature7")["efficiency"].mean()

print("\nMean efficiency by sex:\n", sex_means)
print("\nMean efficiency by help:\n", help_means)
