import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Core columns
cols = [
    "alldeaths",
    "masfem",
    "masfem_mturk",
    "gender_mf",
    "wind",
    "min",
    "category",
    "year",
]

# Keep only rows with needed data
model_df = df[cols].dropna().copy()
model_df["log_deaths"] = np.log1p(model_df["alldeaths"])

# Primary model: log deaths ~ name femininity + storm intensity controls
model1 = smf.ols("log_deaths ~ masfem + wind + min + C(category)", data=model_df).fit()

# Robustness: add year
model2 = smf.ols("log_deaths ~ masfem + wind + min + C(category) + year", data=model_df).fit()

# Robustness: binary gender indicator
model3 = smf.ols("log_deaths ~ gender_mf + wind + min + C(category)", data=model_df).fit()

# Alternative femininity measure (MTurk)
model4 = smf.ols("log_deaths ~ masfem_mturk + wind + min + C(category)", data=model_df).fit()

# Simple correlation (log deaths vs masfem)
cor = model_df["log_deaths"].corr(model_df["masfem"])

results = {
    "n": int(model_df.shape[0]),
    "cor_logdeaths_masfem": float(cor),
    "model1": {
        "coef": float(model1.params["masfem"]),
        "pvalue": float(model1.pvalues["masfem"]),
        "ci_low": float(model1.conf_int().loc["masfem", 0]),
        "ci_high": float(model1.conf_int().loc["masfem", 1]),
        "r2": float(model1.rsquared),
    },
    "model2": {
        "coef": float(model2.params["masfem"]),
        "pvalue": float(model2.pvalues["masfem"]),
        "ci_low": float(model2.conf_int().loc["masfem", 0]),
        "ci_high": float(model2.conf_int().loc["masfem", 1]),
        "r2": float(model2.rsquared),
    },
    "model3": {
        "coef": float(model3.params["gender_mf"]),
        "pvalue": float(model3.pvalues["gender_mf"]),
        "ci_low": float(model3.conf_int().loc["gender_mf", 0]),
        "ci_high": float(model3.conf_int().loc["gender_mf", 1]),
        "r2": float(model3.rsquared),
    },
    "model4": {
        "coef": float(model4.params["masfem_mturk"]),
        "pvalue": float(model4.pvalues["masfem_mturk"]),
        "ci_low": float(model4.conf_int().loc["masfem_mturk", 0]),
        "ci_high": float(model4.conf_int().loc["masfem_mturk", 1]),
        "r2": float(model4.rsquared),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
