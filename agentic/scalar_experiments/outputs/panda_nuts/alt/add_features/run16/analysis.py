import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing values in key columns
key_cols = ["nuts_opened", "seconds", "age", "sex", "help", "chimpanzee"]
df = df.dropna(subset=key_cols).copy()

# Efficiency: nuts opened per second
# Avoid division by zero (if any)
df = df[df["seconds"] > 0].copy()
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Standardize categorical columns
# Normalize 'help' to y/n if mixed case
# Keep only y/n like values
# Some datasets may use 'N' for no.
df["help"] = df["help"].astype(str).str.strip()

# Recode to categorical with baseline
# Use sex as categorical, help as categorical

# Summary statistics
print("Rows:", len(df))
print(df[["efficiency", "nuts_opened", "seconds", "age"]].describe())
print("Sex counts:\n", df["sex"].value_counts())
print("Help counts:\n", df["help"].value_counts())

# OLS regression with robust SEs clustered by chimpanzee
# Model: efficiency ~ age + sex + help

# Make sure categories are treated properly
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df)
ols_res = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})
print("\nOLS (cluster-robust by chimpanzee)\n", ols_res.summary())

# Also check MixedLM with random intercept by chimpanzee (if converges)
try:
    md = smf.mixedlm("efficiency ~ age + C(sex) + C(help)", data=df, groups=df["chimpanzee"])
    mdf = md.fit(reml=False)
    print("\nMixedLM\n", mdf.summary())
except Exception as e:
    print("MixedLM failed:", e)

# Additional check: log efficiency (add small constant)
# In case skewed distribution

df["log_eff"] = np.log(df["efficiency"] + 1e-6)
model_log = smf.ols("log_eff ~ age + C(sex) + C(help)", data=df)
ols_log = model_log.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})
print("\nOLS log efficiency (cluster-robust)\n", ols_log.summary())

# Simple bivariate correlations
print("\nCorrelation age-efficiency:", df["age"].corr(df["efficiency"]))
