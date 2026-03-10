import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}

df = df.rename(columns=col_map)

# Compute efficiency (nuts opened per second)
df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Basic cleaning
# Remove rows with zero or missing duration to avoid divide-by-zero (if any)
df = df.replace([np.inf, -np.inf], np.nan)

# Encode categorical variables
df["sex"] = df["sex"].astype("category")
df["help"] = df["help"].astype("category")

# Fit OLS model
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

# Also fit robust SE (HC3) for safety
robust = model.get_robustcov_results(cov_type="HC3")

# Output key summaries
print("n rows:", len(df))
print("Efficiency summary:", df["efficiency"].describe())
print("OLS params:")
print(model.params)
print("OLS p-values:")
print(model.pvalues)
print("Robust p-values (HC3):")
print(robust.pvalues)
print("R-squared:", model.rsquared)
