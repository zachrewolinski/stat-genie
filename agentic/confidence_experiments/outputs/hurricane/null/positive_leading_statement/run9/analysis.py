import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
num_cols = [
    "masfem",
    "masfem_mturk",
    "gender_mf",
    "wind",
    "min",
    "category",
    "alldeaths",
    "ndam",
    "ndam15",
    "year",
    "elapsedyrs",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Outcome: log deaths (add 1 to handle zeros)
df["log_deaths"] = np.log1p(df["alldeaths"].fillna(0))

# Standardize some predictors for comparability (z-scores)
for c in ["masfem", "wind", "min", "category", "ndam15", "year"]:
    if c in df.columns:
        df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

# Helper to fit OLS with robust SE

def ols(formula, data):
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    return model

# Models
models = {}

# Base: masfem only
models["base"] = ols("log_deaths ~ z_masfem", df)

# Control for storm intensity and damage
models["controls"] = ols(
    "log_deaths ~ z_masfem + z_wind + z_min + z_category + z_ndam15 + z_year",
    df,
)

# Interaction with intensity (wind)
models["interaction_wind"] = ols(
    "log_deaths ~ z_masfem * z_wind + z_min + z_category + z_ndam15 + z_year",
    df,
)

# Interaction with category (proxy for severity)
models["interaction_cat"] = ols(
    "log_deaths ~ z_masfem * z_category + z_wind + z_min + z_ndam15 + z_year",
    df,
)

# Poisson regression for count outcome (with robust SE)
poisson = smf.glm(
    "alldeaths ~ z_masfem + z_wind + z_min + z_category + z_ndam15 + z_year",
    data=df,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

# Negative binomial (overdispersion)
nb = smf.glm(
    "alldeaths ~ z_masfem + z_wind + z_min + z_category + z_ndam15 + z_year",
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
).fit(cov_type="HC3")

# Collect key results
results = {}

def coef_info(model, term):
    if term not in model.params.index:
        return None
    return {
        "coef": float(model.params[term]),
        "se": float(model.bse[term]),
        "p": float(model.pvalues[term]),
    }

results["ols_base"] = coef_info(models["base"], "z_masfem")
results["ols_controls"] = coef_info(models["controls"], "z_masfem")
results["ols_interaction_wind"] = {
    "masfem": coef_info(models["interaction_wind"], "z_masfem"),
    "interaction": coef_info(models["interaction_wind"], "z_masfem:z_wind"),
}
results["ols_interaction_cat"] = {
    "masfem": coef_info(models["interaction_cat"], "z_masfem"),
    "interaction": coef_info(models["interaction_cat"], "z_masfem:z_category"),
}
results["poisson"] = coef_info(poisson, "z_masfem")
results["neg_bin"] = coef_info(nb, "z_masfem")

# Simple correlations
corr = df[["masfem", "alldeaths", "log_deaths", "wind", "category", "ndam15"]].corr()

# Output for inspection
print("N", len(df))
print("Correlation (masfem vs log_deaths)", corr.loc["masfem", "log_deaths"])
print("Correlation (masfem vs alldeaths)", corr.loc["masfem", "alldeaths"])
print(json.dumps(results, indent=2))

# Also print model summaries for the key OLS and Poisson
print("\nOLS controls summary (robust SE):")
print(models["controls"].summary())
print("\nOLS interaction (wind) summary (robust SE):")
print(models["interaction_wind"].summary())
print("\nOLS interaction (category) summary (robust SE):")
print(models["interaction_cat"].summary())
print("\nPoisson summary (robust SE):")
print(poisson.summary())
print("\nNegBin summary (robust SE):")
print(nb.summary())
