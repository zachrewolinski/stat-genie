import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("hurricane.csv")

# Core variables
vars_needed = ["masfem", "gender_mf", "alldeaths", "wind", "min", "category", "year"]

# Keep rows with non-missing for needed variables
analysis_df = df[vars_needed].dropna().copy()

# Outcomes and transforms
analysis_df["log_deaths"] = np.log1p(analysis_df["alldeaths"].astype(float))

# Simple correlations
pearson_r, pearson_p = stats.pearsonr(analysis_df["masfem"], analysis_df["log_deaths"])
spearman_r, spearman_p = stats.spearmanr(analysis_df["masfem"], analysis_df["log_deaths"])

results = {
    "n": int(len(analysis_df)),
    "pearson": {"r": float(pearson_r), "p": float(pearson_p)},
    "spearman": {"r": float(spearman_r), "p": float(spearman_p)},
    "models": {}
}

# OLS models with HC3 robust SEs
models = {
    "m1": "log_deaths ~ masfem",
    "m2": "log_deaths ~ masfem + wind + min + category + year",
    "m3": "log_deaths ~ gender_mf + wind + min + category + year",
}

for key, formula in models.items():
    model = smf.ols(formula, data=analysis_df).fit(cov_type="HC3")
    coef = model.params.get("masfem", model.params.get("gender_mf", np.nan))
    pval = model.pvalues.get("masfem", model.pvalues.get("gender_mf", np.nan))
    results["models"][key] = {
        "formula": formula,
        "coef": float(coef),
        "p": float(pval),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }

# GLM Poisson as a robustness check for count data
try:
    glm_poisson = smf.glm(
        "alldeaths ~ masfem + wind + min + category + year",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")
    # McFadden-like pseudo R^2 using deviance
    pseudo_r2 = np.nan
    if getattr(glm_poisson, "null_deviance", None) not in (None, 0):
        pseudo_r2 = 1 - (glm_poisson.deviance / glm_poisson.null_deviance)
    results["models"]["poisson"] = {
        "formula": "alldeaths ~ masfem + wind + min + category + year (Poisson GLM)",
        "coef": float(glm_poisson.params.get("masfem", np.nan)),
        "p": float(glm_poisson.pvalues.get("masfem", np.nan)),
        "pseudo_r2": float(pseudo_r2),
        "overdispersion": float(glm_poisson.deviance / glm_poisson.df_resid),
    }
except Exception as e:
    results["models"]["poisson_error"] = str(e)

# Negative Binomial GLM as robustness check for overdispersed counts
try:
    glm_nb = smf.glm(
        "alldeaths ~ masfem + wind + min + category + year",
        data=analysis_df,
        family=sm.families.NegativeBinomial(),
    ).fit(cov_type="HC3")
    nb_pseudo_r2 = np.nan
    if getattr(glm_nb, "null_deviance", None) not in (None, 0):
        nb_pseudo_r2 = 1 - (glm_nb.deviance / glm_nb.null_deviance)
    results["models"]["neg_binomial"] = {
        "formula": "alldeaths ~ masfem + wind + min + category + year (NegBin GLM)",
        "coef": float(glm_nb.params.get("masfem", np.nan)),
        "p": float(glm_nb.pvalues.get("masfem", np.nan)),
        "pseudo_r2": float(nb_pseudo_r2),
    }
except Exception as e:
    results["models"]["neg_binomial_error"] = str(e)

# Discrete Negative Binomial (NB2) with estimated dispersion
try:
    nb2_model = smf.negativebinomial(
        "alldeaths ~ masfem + wind + min + category + year",
        data=analysis_df,
    ).fit(disp=0)
    results["models"]["neg_binomial_nb2"] = {
        "formula": "alldeaths ~ masfem + wind + min + category + year (Discrete NB2)",
        "coef": float(nb2_model.params.get("masfem", np.nan)),
        "p": float(nb2_model.pvalues.get("masfem", np.nan)),
        "alpha": float(nb2_model.params.get("alpha", np.nan)),
    }
except Exception as e:
    results["models"]["neg_binomial_nb2_error"] = str(e)

# Save results to a json for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
