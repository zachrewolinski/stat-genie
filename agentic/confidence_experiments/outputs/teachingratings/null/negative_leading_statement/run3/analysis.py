import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
n_rows = len(df)

# Correlations
pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
spearman_r, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

# Simple OLS
m1 = smf.ols("eval ~ beauty", data=df).fit()

# Multivariate OLS with common controls
m2 = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents",
    data=df,
).fit()

# Standardized effect for beauty in models
beauty_std = df["beauty"].std(ddof=1)
eval_std = df["eval"].std(ddof=1)

m1_beta = m1.params["beauty"] * (beauty_std / eval_std)
m2_beta = m2.params["beauty"] * (beauty_std / eval_std)

# Predicted change in eval for 1 SD increase in beauty
m1_sd_effect = m1.params["beauty"] * beauty_std
m2_sd_effect = m2.params["beauty"] * beauty_std

results = {
    "n_rows": n_rows,
    "pearson": {"r": pearson_r, "p": pearson_p},
    "spearman": {"r": spearman_r, "p": spearman_p},
    "m1": {
        "coef_beauty": m1.params["beauty"],
        "se_beauty": m1.bse["beauty"],
        "p_beauty": m1.pvalues["beauty"],
        "r2": m1.rsquared,
        "beta_std": m1_beta,
        "sd_effect": m1_sd_effect,
    },
    "m2": {
        "coef_beauty": m2.params["beauty"],
        "se_beauty": m2.bse["beauty"],
        "p_beauty": m2.pvalues["beauty"],
        "r2": m2.rsquared,
        "beta_std": m2_beta,
        "sd_effect": m2_sd_effect,
    },
}

print(json.dumps(results, indent=2))
