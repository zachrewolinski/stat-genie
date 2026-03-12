import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
print("rows", len(df))
print("columns", df.columns.tolist())

# Focus on relevant columns
cols = [
    "masfem",
    "gender_mf",
    "alldeaths",
    "wind",
    "min",
    "category",
    "ndam",
    "ndam15",
    "year",
]

existing_cols = [c for c in cols if c in df.columns]
print("existing_cols", existing_cols)

sub = df[existing_cols].copy()

# Handle missing values
sub = sub.dropna()

# Log-transform deaths to reduce skew; add 1 for zeros
sub["log_deaths"] = np.log1p(sub["alldeaths"]) if "alldeaths" in sub.columns else np.nan
sub["log_ndam15"] = np.log1p(sub["ndam15"]) if "ndam15" in sub.columns else np.nan
sub["log_ndam"] = np.log1p(sub["ndam"]) if "ndam" in sub.columns else np.nan

print("after_dropna", len(sub))

# Simple correlation between femininity and deaths
if "masfem" in sub.columns and "alldeaths" in sub.columns:
    corr = sub[["masfem", "alldeaths"]].corr().iloc[0,1]
    print("corr_masfem_deaths", corr)

# Compare deaths by gender (male vs female names)
if "gender_mf" in sub.columns and "alldeaths" in sub.columns:
    print("mean_deaths_by_gender")
    print(sub.groupby("gender_mf")["alldeaths"].mean())
    print("median_deaths_by_gender")
    print(sub.groupby("gender_mf")["alldeaths"].median())

# OLS regression: log_deaths ~ masfem + wind + min + category + year
if set(["log_deaths", "masfem", "wind", "min", "category", "year"]).issubset(sub.columns):
    model = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=sub).fit()
    print("ols_log_deaths_masfem")
    print(model.summary().tables[1])

# OLS regression: log_deaths ~ gender_mf + wind + min + category + year
if set(["log_deaths", "gender_mf", "wind", "min", "category", "year"]).issubset(sub.columns):
    model2 = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=sub).fit()
    print("ols_log_deaths_gender")
    print(model2.summary().tables[1])

# Robustness: include log damages if available (proxy for exposure/wealth)
if set(["log_deaths", "masfem", "wind", "min", "category", "year", "log_ndam15"]).issubset(sub.columns):
    model3 = smf.ols("log_deaths ~ masfem + wind + min + category + year + log_ndam15", data=sub).fit()
    print("ols_log_deaths_masfem_ndam15")
    print(model3.summary().tables[1])

# Negative binomial on raw deaths (count model)
if set(["alldeaths", "masfem", "wind", "min", "category", "year"]).issubset(sub.columns):
    try:
        nb = smf.glm("alldeaths ~ masfem + wind + min + category + year", data=sub, family=sm.families.NegativeBinomial()).fit()
        print("nb_deaths_masfem")
        print(nb.summary().tables[1])
    except Exception as e:
        print("nb_failed", e)

# Negative binomial with gender
if set(["alldeaths", "gender_mf", "wind", "min", "category", "year"]).issubset(sub.columns):
    try:
        nb2 = smf.glm("alldeaths ~ gender_mf + wind + min + category + year", data=sub, family=sm.families.NegativeBinomial()).fit()
        print("nb_deaths_gender")
        print(nb2.summary().tables[1])
    except Exception as e:
        print("nb_failed_gender", e)

