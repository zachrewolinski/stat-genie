import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic cleaning
# Keep rows with required fields
required_cols = ["alldeaths", "masfem", "wind", "min", "category", "year"]
df = _df.dropna(subset=required_cols).copy()

# Transform outcome to handle zeros and skew
# log1p keeps zeros and reduces skew in fatalities
_df["log_deaths"] = np.log1p(_df["alldeaths"])
df["log_deaths"] = np.log1p(df["alldeaths"])

print("Rows used:", len(df))
print("Deaths summary (raw):")
print(df["alldeaths"].describe())
print("Deaths summary (log1p):")
print(df["log_deaths"].describe())

# Simple correlations
corr_masfem = df[["masfem", "log_deaths"]].corr().iloc[0, 1]
corr_gender = df[["gender_mf", "log_deaths"]].corr().iloc[0, 1]
print(f"Correlation (masfem vs log deaths): {corr_masfem:.3f}")
print(f"Correlation (gender_mf vs log deaths): {corr_gender:.3f}")

# Regression models
# 1) Unadjusted
m1 = smf.ols("log_deaths ~ masfem", data=df).fit(cov_type="HC3")
print("\nModel 1: log_deaths ~ masfem")
print(m1.summary().tables[1])

# 2) Adjust for storm intensity and time
m2 = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=df).fit(cov_type="HC3")
print("\nModel 2: log_deaths ~ masfem + wind + min + category + year")
print(m2.summary().tables[1])

# 3) Use binary gender indicator instead of continuous scale
m3 = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=df).fit(cov_type="HC3")
print("\nModel 3: log_deaths ~ gender_mf + wind + min + category + year")
print(m3.summary().tables[1])

# Extract key coefficients for quick interpretation
for name, model in [("m1", m1), ("m2", m2)]:
    coef = model.params.get("masfem", np.nan)
    pval = model.pvalues.get("masfem", np.nan)
    print(f"{name} masfem coef: {coef:.4f}, p-value: {pval:.4f}")

coef_gender = m3.params.get("gender_mf", np.nan)
pval_gender = m3.pvalues.get("gender_mf", np.nan)
print(f"m3 gender_mf coef: {coef_gender:.4f}, p-value: {pval_gender:.4f}")
