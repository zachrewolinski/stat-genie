import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic prep
_df = _df.copy()
_df["log_deaths"] = np.log1p(_df["alldeaths"])
_df["log_ndam15"] = np.log1p(_df["ndam15"])

# Standardize some predictors for comparability (optional for interpretation)
# We'll keep raw for reporting, but compute standardized effect size too.
_df["masfem_z"] = (_df["masfem"] - _df["masfem"].mean()) / _df["masfem"].std(ddof=0)

results = {}

# 1) Simple correlation between femininity and deaths
corr = _df[["masfem", "alldeaths"]].corr().iloc[0,1]
results["corr_masfem_deaths"] = corr

# 2) OLS on log deaths: unadjusted
m1 = smf.ols("log_deaths ~ masfem", data=_df).fit(cov_type="HC3")
results["m1"] = {
    "coef": m1.params["masfem"],
    "pval": m1.pvalues["masfem"],
    "r2": m1.rsquared,
}

# 3) Adjust for storm intensity: wind, min pressure, category
# These are common controls; category and wind are correlated, but include to follow prior analyses.
# Use robust SEs.
m2 = smf.ols("log_deaths ~ masfem + wind + min + category", data=_df).fit(cov_type="HC3")
results["m2"] = {
    "coef": m2.params["masfem"],
    "pval": m2.pvalues["masfem"],
    "r2": m2.rsquared,
}

# 4) Add year (or elapsedyrs) to control for temporal changes
m3 = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=_df).fit(cov_type="HC3")
results["m3"] = {
    "coef": m3.params["masfem"],
    "pval": m3.pvalues["masfem"],
    "r2": m3.rsquared,
}

# 5) Alternative using binary gender indicator
m4 = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=_df).fit(cov_type="HC3")
results["m4"] = {
    "coef": m4.params["gender_mf"],
    "pval": m4.pvalues["gender_mf"],
    "r2": m4.rsquared,
}

# 6) Poisson GLM for deaths (count), with overdispersion check
# We use Poisson with robust SEs as a sensitivity check.
try:
    glm = smf.glm(
        "alldeaths ~ masfem + wind + min + category + year",
        data=_df,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")
    results["glm"] = {
        "coef": glm.params["masfem"],
        "pval": glm.pvalues["masfem"],
    }
except Exception as e:
    results["glm_error"] = str(e)

# 6b) Negative Binomial as an overdispersion-robust check
try:
    nb = smf.glm(
        "alldeaths ~ masfem + wind + min + category + year",
        data=_df,
        family=sm.families.NegativeBinomial(),
    ).fit(cov_type="HC3")
    results["nb"] = {
        "coef": nb.params["masfem"],
        "pval": nb.pvalues["masfem"],
    }
except Exception as e:
    results["nb_error"] = str(e)

# 7) Standardized effect size in adjusted model
m2_z = smf.ols("log_deaths ~ masfem_z + wind + min + category", data=_df).fit(cov_type="HC3")
results["m2_z"] = {
    "coef": m2_z.params["masfem_z"],
    "pval": m2_z.pvalues["masfem_z"],
}

# Save results for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
