import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Map features to readable names
cols = {
    "feature2": "year",
    "feature4": "masfem",
    "feature5": "min_pressure",
    "feature6": "female",
    "feature7": "category",
    "feature8": "deaths",
    "feature9": "damage2013",
    "feature13": "wind",
}

for k, v in cols.items():
    if k in df.columns:
        df[v] = df[k]

# Basic cleaning
analysis_df = df[["masfem", "female", "category", "wind", "min_pressure", "deaths", "year"]].copy()
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()

analysis_df["log_deaths"] = np.log1p(analysis_df["deaths"].astype(float))

results = {}

def fit_ols(formula_name, y, X):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc)
    res = model.fit(cov_type="HC3")
    results[formula_name] = {
        "n": int(res.nobs),
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "rsquared": float(res.rsquared),
    }
    return res

# Model 1: unadjusted femininity
fit_ols(
    "log_deaths_masfem",
    analysis_df["log_deaths"],
    analysis_df[["masfem"]],
)

# Model 2: adjusted for intensity
fit_ols(
    "log_deaths_masfem_controls",
    analysis_df["log_deaths"],
    analysis_df[["masfem", "category", "wind", "min_pressure"]],
)

# Model 3: binary female with controls
fit_ols(
    "log_deaths_female_controls",
    analysis_df["log_deaths"],
    analysis_df[["female", "category", "wind", "min_pressure"]],
)

# Model 4: add year trend
fit_ols(
    "log_deaths_masfem_controls_year",
    analysis_df["log_deaths"],
    analysis_df[["masfem", "category", "wind", "min_pressure", "year"]],
)

# Correlation between femininity and deaths
results["corr"] = {
    "masfem_log_deaths": float(np.corrcoef(analysis_df["masfem"], analysis_df["log_deaths"])[0, 1]),
    "female_log_deaths": float(np.corrcoef(analysis_df["female"], analysis_df["log_deaths"])[0, 1]),
}

# Effect size: expected multiplier for 1 SD increase in masfem (log scale)
masfem_sd = analysis_df["masfem"].std()
coef = results["log_deaths_masfem_controls"]["params"].get("masfem", np.nan)
results["effect_1sd_masfem_controls"] = {
    "masfem_sd": float(masfem_sd),
    "log_change": float(coef * masfem_sd),
    "death_multiplier": float(np.exp(coef * masfem_sd)),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
