import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv("teachingratings.csv")

# Basic stats
n = len(df)

# Pearson correlation
corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Multivariate controls
# Use categorical controls and continuous controls
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
    "C(tenure) + C(division) + C(credits) + students + allstudents"
)

# Cluster-robust SE by professor (to account for repeated instructors)
model_full = smf.ols(formula, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["prof"]}
)

# Standardized effect (per 1 SD) using simple model coefficient
beauty_sd = df["beauty"].std(ddof=1)

simple_beta = model_simple.params["beauty"]
simple_beta_se = model_simple.bse["beauty"]

full_beta = model_full.params["beauty"]
full_beta_se = model_full.bse["beauty"]

# 95% CI
simple_ci = model_simple.conf_int().loc["beauty"].tolist()
full_ci = model_full.conf_int().loc["beauty"].tolist()

# Effect of 1 SD beauty in eval units
simple_sd_effect = simple_beta * beauty_sd
full_sd_effect = full_beta * beauty_sd

results = {
    "n": n,
    "corr": corr,
    "corr_p": corr_p,
    "simple": {
        "beta": simple_beta,
        "se": simple_beta_se,
        "p": model_simple.pvalues["beauty"],
        "ci": simple_ci,
        "sd_effect": simple_sd_effect,
        "r2": model_simple.rsquared,
    },
    "full": {
        "beta": full_beta,
        "se": full_beta_se,
        "p": model_full.pvalues["beauty"],
        "ci": full_ci,
        "sd_effect": full_sd_effect,
        "r2": model_full.rsquared,
    },
}

print(json.dumps(results, indent=2))
