import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Outcome: log deaths (skewed counts)
df = df.copy()
df["log_deaths"] = np.log1p(df["alldeaths"])

# Core predictors
predictors_basic = ["masfem"]
controls = ["wind", "min", "category"]
controls_year = controls + ["year"]

# Helper to fit OLS with robust SE

def fit_ols(formula, data):
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    return model

results = {}

# Model 1: unadjusted
m1 = fit_ols("log_deaths ~ masfem", df)
results["m1"] = {
    "coef": float(m1.params["masfem"]),
    "pval": float(m1.pvalues["masfem"]),
    "r2": float(m1.rsquared),
}

# Model 2: severity controls
m2 = fit_ols("log_deaths ~ masfem + wind + min + category", df)
results["m2"] = {
    "coef": float(m2.params["masfem"]),
    "pval": float(m2.pvalues["masfem"]),
    "r2": float(m2.rsquared),
}

# Model 3: add year to capture trends
m3 = fit_ols("log_deaths ~ masfem + wind + min + category + year", df)
results["m3"] = {
    "coef": float(m3.params["masfem"]),
    "pval": float(m3.pvalues["masfem"]),
    "r2": float(m3.rsquared),
}

# Alternative femininity measure (MTurk)
m4 = fit_ols("log_deaths ~ masfem_mturk + wind + min + category", df)
results["m4"] = {
    "coef": float(m4.params["masfem_mturk"]),
    "pval": float(m4.pvalues["masfem_mturk"]),
    "r2": float(m4.rsquared),
}

# Binary gender indicator model
m5 = fit_ols("log_deaths ~ gender_mf + wind + min + category", df)
results["m5"] = {
    "coef": float(m5.params["gender_mf"]),
    "pval": float(m5.pvalues["gender_mf"]),
    "r2": float(m5.rsquared),
}

# Simple correlations
corr_masfem = float(df["masfem"].corr(df["log_deaths"]))
corr_mturk = float(df["masfem_mturk"].corr(df["log_deaths"]))
results["corr"] = {
    "masfem_log_deaths": corr_masfem,
    "masfem_mturk_log_deaths": corr_mturk,
}

# Group difference by gender_mf
male = df.loc[df["gender_mf"] == 0, "log_deaths"]
female = df.loc[df["gender_mf"] == 1, "log_deaths"]
results["means"] = {
    "log_deaths_male": float(male.mean()),
    "log_deaths_female": float(female.mean()),
}

print(json.dumps(results, indent=2))
