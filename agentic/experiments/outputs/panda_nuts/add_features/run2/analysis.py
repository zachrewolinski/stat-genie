import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning for relevant columns
for col in ["sex", "help"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Drop rows with missing predictors or outcomes
needed = ["nuts_opened", "seconds", "age", "sex", "help"]
df_clean = df.dropna(subset=needed).copy()

# Remove any rows with non-positive session time (shouldn't happen, but safe)
df_clean = df_clean[df_clean["seconds"] > 0].copy()

# Efficiency: nuts opened per second
# Add a small epsilon to avoid divide-by-zero (not needed here but safe)
df_clean["efficiency"] = df_clean["nuts_opened"] / df_clean["seconds"]

# OLS on efficiency with robust SEs
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df_clean).fit(cov_type="HC3")

# Poisson model on counts with log(seconds) offset (models rate)
df_clean["log_seconds"] = np.log(df_clean["seconds"])
poisson_model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df_clean,
    family=sm.families.Poisson(),
    offset=df_clean["log_seconds"],
).fit(cov_type="HC3")

print("Rows used:", len(df_clean))
print("\nOLS (efficiency) with robust SEs:\n")
print(ols_model.summary())
print("\nPoisson (rate) with robust SEs:\n")
print(poisson_model.summary())

# Save key results for use in conclusion
results = {
    "ols_pvalues": ols_model.pvalues.to_dict(),
    "poisson_pvalues": poisson_model.pvalues.to_dict(),
    "ols_params": ols_model.params.to_dict(),
    "poisson_params": poisson_model.params.to_dict(),
}

pd.Series(results["ols_pvalues"]).to_csv("ols_pvalues.csv")
pd.Series(results["poisson_pvalues"]).to_csv("poisson_pvalues.csv")
