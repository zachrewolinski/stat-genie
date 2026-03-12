import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Keep relevant columns; ignore unrelated added features.
cols = ["alldeaths", "masfem", "gender_mf", "wind", "min", "category", "year", "ndam15", "ndam"]
cols = [c for c in cols if c in df.columns]
sub = df[cols].copy()

# Coerce to numeric where applicable
for c in cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# Drop rows with missing values in core variables
core = [c for c in ["alldeaths", "masfem", "wind", "min", "category"] if c in sub.columns]
sub = sub.dropna(subset=core)

# Outcome transformations
sub["log_deaths"] = np.log1p(sub["alldeaths"])

results = {}

# OLS with robust SEs
formula = "log_deaths ~ masfem + wind + min + category"
ols = smf.ols(formula, data=sub).fit(cov_type="HC3")
results["ols"] = {
    "n": int(ols.nobs),
    "coef_masfem": float(ols.params.get("masfem", np.nan)),
    "p_masfem": float(ols.pvalues.get("masfem", np.nan)),
    "ci_masfem": [float(x) for x in ols.conf_int().loc["masfem"].tolist()],
    "r2": float(ols.rsquared),
}

# Negative binomial GLM (counts) with same predictors
try:
    nb = smf.glm(
        "alldeaths ~ masfem + wind + min + category",
        data=sub,
        family=sm.families.NegativeBinomial(),
    ).fit()
    results["negbin"] = {
        "n": int(nb.nobs),
        "coef_masfem": float(nb.params.get("masfem", np.nan)),
        "p_masfem": float(nb.pvalues.get("masfem", np.nan)),
        "ci_masfem": [float(x) for x in nb.conf_int().loc["masfem"].tolist()],
    }
except Exception as e:
    results["negbin_error"] = str(e)

# Simple bivariate correlation for context
corr = sub[["alldeaths", "masfem"]].corr().iloc[0,1]
results["corr_deaths_masfem"] = float(corr)

# Additional model using gender_mf indicator if present
if "gender_mf" in sub.columns:
    gdf = sub.dropna(subset=["gender_mf"])
    ols_g = smf.ols("log_deaths ~ gender_mf + wind + min + category", data=gdf).fit(cov_type="HC3")
    results["ols_gender"] = {
        "n": int(ols_g.nobs),
        "coef_gender_female": float(ols_g.params.get("gender_mf", np.nan)),
        "p_gender_female": float(ols_g.pvalues.get("gender_mf", np.nan)),
        "ci_gender_female": [float(x) for x in ols_g.conf_int().loc["gender_mf"].tolist()],
        "r2": float(ols_g.rsquared),
    }

print(json.dumps(results, indent=2))
