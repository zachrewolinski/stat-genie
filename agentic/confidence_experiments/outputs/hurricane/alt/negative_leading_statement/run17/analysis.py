import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic cleaning
_df = _df.copy()
_df["log_deaths"] = np.log1p(_df["alldeaths"])
_df["log_ndam15"] = np.log1p(_df["ndam15"])


def extract_param(res, name: str):
    names = list(res.model.exog_names)
    if name in names:
        idx = names.index(name)
        return float(res.params[idx]), float(res.pvalues[idx])
    return float("nan"), float("nan")


def get_glm_robust(res, cov_type="HC3"):
    # statsmodels GLMResults exposes a protected method for robust covariances
    try:
        robust = res._get_robustcov_results(cov_type=cov_type)
        return robust if robust is not None else res
    except Exception:
        return res


# Build model specs (OLS on log deaths)
specs = {
    "bivariate": "log_deaths ~ masfem",
    "controls_intensity": "log_deaths ~ masfem + wind + min + category",
    "controls_intensity_year": "log_deaths ~ masfem + wind + min + category + year",
    "controls_intensity_year_damage": "log_deaths ~ masfem + wind + min + category + year + log_ndam15",
}

results = {}
for name, formula in specs.items():
    model = smf.ols(formula, data=_df).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    coef, pval = extract_param(robust, "masfem")
    results[name] = {
        "coef": coef,
        "pval": pval,
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }

# Poisson GLM on counts (with robust SE)
poisson_formula = "alldeaths ~ masfem + wind + min + category + year"
poisson_model = smf.glm(poisson_formula, data=_df, family=sm.families.Poisson()).fit()
poisson_robust = get_glm_robust(poisson_model, cov_type="HC3")
poisson_coef, poisson_pval = extract_param(poisson_robust, "masfem")
results["poisson_counts_hc3"] = {
    "coef": poisson_coef,
    "pval": poisson_pval,
    "n": int(poisson_model.nobs),
}

# Negative binomial GLM (alpha fixed=1) with robust SE
nb_model_glm = smf.glm(poisson_formula, data=_df, family=sm.families.NegativeBinomial()).fit()
nb_robust = get_glm_robust(nb_model_glm, cov_type="HC3")
nb_coef, nb_pval = extract_param(nb_robust, "masfem")
results["negbin_glm_hc3"] = {
    "coef": nb_coef,
    "pval": nb_pval,
    "n": int(nb_model_glm.nobs),
}

# Negative binomial MLE (alpha estimated)
try:
    nb_mle = smf.negativebinomial(poisson_formula, data=_df).fit(disp=False)
    nb_mle_robust = nb_mle.get_robustcov_results(cov_type="HC3")
    nb_mle_coef, nb_mle_pval = extract_param(nb_mle_robust, "masfem")
    results["negbin_mle_hc3"] = {
        "coef": nb_mle_coef,
        "pval": nb_mle_pval,
        "n": int(nb_mle.nobs),
    }
except Exception as exc:
    results["negbin_mle_hc3"] = {
        "error": str(exc)
    }

# Correlations
corr_raw = _df[["masfem", "alldeaths"]].corr().iloc[0, 1]
corr_log = _df[["masfem", "log_deaths"]].corr().iloc[0, 1]
results["correlations"] = {
    "corr_raw": float(corr_raw),
    "corr_log": float(corr_log),
}

# Overdispersion check for counts
mean_deaths = _df["alldeaths"].mean()
var_deaths = _df["alldeaths"].var(ddof=1)
results["overdispersion"] = {
    "mean_deaths": float(mean_deaths),
    "var_deaths": float(var_deaths),
    "var_to_mean": float(var_deaths / mean_deaths) if mean_deaths > 0 else float("nan"),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
