import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Keep relevant columns
cols = ["nuts_opened", "seconds", "age", "sex", "help"]
df = df[cols].copy()

# Drop missing or invalid values
# Ensure seconds > 0 to compute rates/offsets
mask = df["seconds"].notna() & (df["seconds"] > 0)
mask &= df["nuts_opened"].notna()
mask &= df["age"].notna() & df["sex"].notna() & df["help"].notna()
df = df[mask].copy()

# Compute efficiency (nuts per second)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Encode categorical variables
# Use categorical with explicit ordering not necessary

# Poisson GLM with offset for time
formula = "nuts_opened ~ age + C(sex) + C(help)"
poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit()

# Check overdispersion
dispersion = poisson_model.deviance / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan

# Negative binomial GLM as robustness
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df["seconds"])
).fit()

# OLS on efficiency for a complementary check
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

print("Rows used:", len(df))
print("Efficiency summary:")
print(df["efficiency"].describe())

print("\nPoisson GLM (rate model with offset):")
print(poisson_model.summary())
print("Dispersion (deviance/df_resid):", dispersion)

print("\nNegative Binomial GLM:")
print(nb_model.summary())

print("\nOLS on efficiency (HC3 robust SE):")
print(ols_model.summary())

# Convenience: show coefficients and p-values
print("\nPoisson params:")
print(poisson_model.params)
print("\nPoisson p-values:")
print(poisson_model.pvalues)

print("\nNB params:")
print(nb_model.params)
print("\nNB p-values:")
print(nb_model.pvalues)

print("\nOLS params:")
print(ols_model.params)
print("\nOLS p-values:")
print(ols_model.pvalues)
