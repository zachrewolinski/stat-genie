import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# Efficiency: nuts opened per minute
# If seconds is zero (shouldn't be), avoid division
seconds = df["seconds"].replace(0, np.nan)
df["nuts_per_min"] = df["nuts_opened"] / (seconds / 60.0)

# Drop rows with missing efficiency
df_eff = df.dropna(subset=["nuts_per_min"]).copy()

# Model 1: OLS on efficiency
m1 = smf.ols("nuts_per_min ~ age + sex + help", data=df_eff).fit()

# Model 2: OLS on efficiency with hammer control
m2 = smf.ols("nuts_per_min ~ age + sex + help + hammer", data=df_eff).fit()

# Model 3: Poisson regression for counts with log(seconds) offset
# Use robust (HC3) covariance for Poisson is not typical; use default and also robust as check
m3 = smf.glm(
    "nuts_opened ~ age + sex + help",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit()

# Model 4: Poisson with hammer control
m4 = smf.glm(
    "nuts_opened ~ age + sex + help + hammer",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit()

# Summaries to print
print("N rows:", len(df))
print("Efficiency stats (nuts/min):")
print(df_eff["nuts_per_min"].describe())

print("\nOLS model 1 coefficients:")
print(m1.params)
print("OLS model 1 p-values:")
print(m1.pvalues)
print("OLS model 1 R2:", m1.rsquared)

print("\nOLS model 2 coefficients:")
print(m2.params)
print("OLS model 2 p-values:")
print(m2.pvalues)
print("OLS model 2 R2:", m2.rsquared)

print("\nPoisson model 3 coefficients:")
print(m3.params)
print("Poisson model 3 p-values:")
print(m3.pvalues)

print("\nPoisson model 4 coefficients:")
print(m4.params)
print("Poisson model 4 p-values:")
print(m4.pvalues)

# Also report average efficiency by sex and help for context
print("\nGroup means (nuts/min):")
print(df_eff.groupby("sex")["nuts_per_min"].mean())
print(df_eff.groupby("help")["nuts_per_min"].mean())

# Correlation of age with efficiency
print("\nCorrelation age vs nuts/min:", df_eff["age"].corr(df_eff["nuts_per_min"]))
