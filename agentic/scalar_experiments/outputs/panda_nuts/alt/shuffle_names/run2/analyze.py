import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Map columns to inferred meanings
# age -> age
# nuts_opened column (m/f) -> sex
# sex column (wood/Q/G/...) -> hammer type (unused)
# help column -> nuts opened count
# chimpanzee column -> seconds
# seconds column (y/N) -> help (received help)

df = df.rename(columns={
    "nuts_opened": "sex",
    "help": "nuts_opened",
    "chimpanzee": "seconds",
    "seconds": "help",
    "sex": "hammer_type",
})

# Clean help to boolean
help_map = {"y": 1, "Y": 1, "yes": 1, "Yes": 1, "N": 0, "n": 0, "no": 0, "No": 0}
df["help"] = df["help"].map(help_map)

# Ensure numeric
for col in ["age", "nuts_opened", "seconds"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values
model_df = df.dropna(subset=["age", "sex", "help", "nuts_opened", "seconds"]).copy()

# Efficiency
model_df["efficiency"] = model_df["nuts_opened"] / model_df["seconds"]

# Basic summaries
summary = {
    "n_rows": len(model_df),
    "eff_mean": model_df["efficiency"].mean(),
    "eff_std": model_df["efficiency"].std(),
}
print("SUMMARY", summary)

# OLS on efficiency
ols = smf.ols("efficiency ~ age + C(sex) + help", data=model_df).fit()
print("OLS")
print(ols.summary())

# Poisson model for counts with offset (seconds)
# Avoid zero seconds
poisson_df = model_df[model_df["seconds"] > 0].copy()
poisson = smf.glm("nuts_opened ~ age + C(sex) + help", data=poisson_df,
                 family=sm.families.Poisson(), offset=np.log(poisson_df["seconds"])).fit()
print("POISSON")
print(poisson.summary())

# Overdispersion check
mu = poisson.predict()
resid = poisson_df["nuts_opened"] - mu
od_ratio = (resid**2).sum() / poisson.df_resid
print("OVERDISPERSION_RATIO", od_ratio)

# Negative binomial if overdispersed
nb = smf.glm("nuts_opened ~ age + C(sex) + help", data=poisson_df,
             family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(poisson_df["seconds"])).fit()
print("NEGATIVE_BINOMIAL")
print(nb.summary())

# Effect sizes for help (rate ratio)
for model, name in [(poisson, "poisson"), (nb, "nb")]:
    coef = model.params.get("help", np.nan)
    se = model.bse.get("help", np.nan)
    rr = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    print(name, "help_rate_ratio", rr, "CI", ci_low, ci_high)

# Age effect per year (rate ratio)
for model, name in [(poisson, "poisson"), (nb, "nb")]:
    coef = model.params.get("age", np.nan)
    se = model.bse.get("age", np.nan)
    rr = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    print(name, "age_rate_ratio", rr, "CI", ci_low, ci_high)

# Sex effects relative to baseline
# Provide estimates for female vs male if needed
print("SEX_LEVELS", model_df["sex"].unique())
