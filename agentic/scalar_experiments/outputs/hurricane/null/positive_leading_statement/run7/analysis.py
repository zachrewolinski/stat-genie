import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic summaries
summary = {
    "rows": len(_df),
    "masfem_min": float(_df["masfem"].min()),
    "masfem_max": float(_df["masfem"].max()),
    "alldeaths_min": int(_df["alldeaths"].min()),
    "alldeaths_max": int(_df["alldeaths"].max()),
}

# Prepare outcome: log deaths + 1 to handle zeros
_df = _df.copy()
_df["log_deaths"] = np.log1p(_df["alldeaths"].astype(float))

# OLS with controls for intensity/severity
# Controls: wind, min pressure, category, ndam15 (damage proxy), year
# Use robust (HC3) standard errors due to heteroskedasticity
formula = "log_deaths ~ masfem + wind + min + category + ndam15 + year"
model = smf.ols(formula, data=_df).fit(cov_type="HC3")

# Also test binary gender indicator for robustness
formula_gender = "log_deaths ~ gender_mf + wind + min + category + ndam15 + year"
model_gender = smf.ols(formula_gender, data=_df).fit(cov_type="HC3")

# Simple correlation
corr = _df[["masfem", "log_deaths"]].corr().iloc[0,1]

results = {
    "summary": summary,
    "corr_masfem_log_deaths": float(corr),
    "masfem_coef": float(model.params.get("masfem", np.nan)),
    "masfem_p": float(model.pvalues.get("masfem", np.nan)),
    "masfem_ci": [float(x) for x in model.conf_int().loc["masfem"].tolist()],
    "model_r2": float(model.rsquared),
    "gender_coef": float(model_gender.params.get("gender_mf", np.nan)),
    "gender_p": float(model_gender.pvalues.get("gender_mf", np.nan)),
    "gender_ci": [float(x) for x in model_gender.conf_int().loc["gender_mf"].tolist()],
    "gender_r2": float(model_gender.rsquared),
}

print(json.dumps(results, indent=2))
