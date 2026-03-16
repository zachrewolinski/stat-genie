import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Define variables
fem = df["feature4"]  # masculinity-femininity index (higher = more feminine)
mturk = df["feature12"]
sex_bin = df["feature6"]  # 0 male, 1 female

deaths = df["feature8"]
log_deaths = np.log1p(deaths)

controls = df[["feature7", "feature5", "feature13", "feature2"]].copy()
controls.columns = ["category", "min_pressure", "max_wind", "year"]

# Correlations
pearson_fem = stats.pearsonr(fem, log_deaths)
spearman_fem = stats.spearmanr(fem, log_deaths)

pearson_mturk = stats.pearsonr(mturk, log_deaths)
spearman_mturk = stats.spearmanr(mturk, log_deaths)

# OLS models

def fit_ols(y, x, name):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return {
        "name": name,
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "r2": model.rsquared,
        "n": int(model.nobs),
    }

models = []
models.append(
    fit_ols(
        log_deaths,
        pd.concat([fem.rename("femininity"), controls], axis=1),
        "log_deaths ~ femininity + controls",
    )
)
models.append(
    fit_ols(
        log_deaths,
        pd.concat([mturk.rename("mturk_fem"), controls], axis=1),
        "log_deaths ~ mturk_fem + controls",
    )
)
models.append(
    fit_ols(
        log_deaths,
        pd.concat([sex_bin.rename("female"), controls], axis=1),
        "log_deaths ~ female + controls",
    )
)

# Store results for easy inspection
results = {
    "correlations": {
        "femininity": {
            "pearson_r": pearson_fem[0],
            "pearson_p": pearson_fem[1],
            "spearman_r": spearman_fem[0],
            "spearman_p": spearman_fem[1],
        },
        "mturk_fem": {
            "pearson_r": pearson_mturk[0],
            "pearson_p": pearson_mturk[1],
            "spearman_r": spearman_mturk[0],
            "spearman_p": spearman_mturk[1],
        },
    },
    "models": models,
}

print(json.dumps(results, indent=2))
