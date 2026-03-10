import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Focus on relevant columns
cols = ["nuts_opened", "seconds", "age", "sex", "help"]
sub = df[cols].copy()

# Clean / standardize
sub = sub.dropna()
sub = sub[sub["seconds"] > 0]

# Normalize categories
sub["sex"] = sub["sex"].astype(str).str.strip().str.lower()
sub["help"] = sub["help"].astype(str).str.strip().str.lower()

# Keep only expected categories for clarity
# (leave as-is if others exist)

# Efficiency measures
sub["rate_per_sec"] = sub["nuts_opened"] / sub["seconds"]
sub["rate_per_min"] = sub["rate_per_sec"] * 60

print("Rows used:", len(sub))
print("Sex values:", sub["sex"].unique())
print("Help values:", sub["help"].unique())
print(sub[["nuts_opened", "seconds", "age", "rate_per_min"]].describe())

# Poisson GLM with exposure offset
poisson = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=sub,
    family=sm.families.Poisson(),
    offset=np.log(sub["seconds"])
).fit()

print("\nPoisson GLM summary (count w/ log(seconds) offset):")
print(poisson.summary())

# Overdispersion check
pearson_chi2 = poisson.pearson_chi2
pearson_df = poisson.df_resid
ratio = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan
print(f"\nOverdispersion ratio (Pearson chi2 / df): {ratio:.3f}")

# Negative binomial GLM for robustness
nb = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=sub,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(sub["seconds"])
).fit()

print("\nNegative Binomial GLM summary (alpha=1.0):")
print(nb.summary())

# OLS on log(rate_per_min + small constant) for sensitivity
sub["log_rate_per_min"] = np.log(sub["rate_per_min"] + 1e-6)
ols = smf.ols("log_rate_per_min ~ age + C(sex) + C(help)", data=sub).fit(cov_type="HC3")
print("\nOLS on log(rate_per_min) with HC3 robust SEs:")
print(ols.summary())
