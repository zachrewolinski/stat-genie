import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Normalize categorical values to lowercase for consistency
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Efficiency: nuts opened per second
# Avoid division by zero though seconds min is 2.5

df["efficiency"] = df["nuts_opened"] / df["seconds"]

print("Rows:", len(df))
print("Efficiency summary:")
print(df["efficiency"].describe())

# OLS regression on efficiency
ols = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")
print("\nOLS on efficiency (robust HC3):")
print(ols.summary())

# Poisson regression on nuts_opened with offset log(seconds)
# This models rate per second
# Use GLM Poisson; check overdispersion

df["log_seconds"] = np.log(df["seconds"])
poisson = smf.glm("nuts_opened ~ age + C(sex) + C(help)", data=df,
                 family=sm.families.Poisson(), offset=df["log_seconds"]).fit()
print("\nPoisson rate model (offset log seconds):")
print(poisson.summary())

# Overdispersion check: Pearson chi2 / df_resid
pearson_chi2 = poisson.pearson_chi2
ratio = pearson_chi2 / poisson.df_resid
print("\nPoisson overdispersion ratio (Pearson chi2 / df_resid):", ratio)

# Negative Binomial regression if overdispersion is high
try:
    nb = smf.glm("nuts_opened ~ age + C(sex) + C(help)", data=df,
                 family=sm.families.NegativeBinomial(alpha=1.0),
                 offset=df["log_seconds"]).fit()
    print("\nNegBin rate model (offset log seconds):")
    print(nb.summary())
except Exception as e:
    print("\nNegBin fit failed:", e)

# Correlations
print("\nCorrelation age vs efficiency:", df["age"].corr(df["efficiency"]))

# Group means
print("\nEfficiency by sex:")
print(df.groupby("sex")["efficiency"].agg(["mean","median","count"]))

print("\nEfficiency by help:")
print(df.groupby("help")["efficiency"].agg(["mean","median","count"]))

