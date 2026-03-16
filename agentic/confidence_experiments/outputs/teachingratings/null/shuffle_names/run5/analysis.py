import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Ensure categorical columns are treated as categories
cat_cols = ["eval", "tenure", "prof", "native", "gender", "credits"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Core variables
beauty = df["beauty"].astype(float)
ratings = df["allstudents"].astype(float)

# Summary stats
n = len(df)
beauty_mean = beauty.mean()
beauty_sd = beauty.std(ddof=1)
ratings_mean = ratings.mean()
ratings_sd = ratings.std(ddof=1)

# Pearson correlation
corr_r, corr_p = stats.pearsonr(beauty, ratings)

# Simple OLS
model_simple = smf.ols("allstudents ~ beauty", data=df).fit(cov_type="HC3")

# OLS with controls (exclude id-like columns)
controls = [
    "age",
    "C(eval)",
    "C(tenure)",
    "C(prof)",
    "C(native)",
    "C(gender)",
    "C(credits)",
    "rownames",
    "minority",
]
formula_controls = "allstudents ~ beauty + " + " + ".join(controls)
model_controls = smf.ols(formula_controls, data=df).fit(cov_type="HC3")

# Extract effect sizes
coef_simple = model_simple.params["beauty"]
se_simple = model_simple.bse["beauty"]
p_simple = model_simple.pvalues["beauty"]
ci_simple = model_simple.conf_int().loc["beauty"].tolist()

coef_ctrl = model_controls.params["beauty"]
se_ctrl = model_controls.bse["beauty"]
p_ctrl = model_controls.pvalues["beauty"]
ci_ctrl = model_controls.conf_int().loc["beauty"].tolist()

# Standardized effect (beta) using SDs
std_effect_simple = coef_simple * (beauty_sd / ratings_sd)
std_effect_ctrl = coef_ctrl * (beauty_sd / ratings_sd)

results = {
    "n": n,
    "beauty_mean": beauty_mean,
    "beauty_sd": beauty_sd,
    "ratings_mean": ratings_mean,
    "ratings_sd": ratings_sd,
    "corr_r": corr_r,
    "corr_p": corr_p,
    "simple": {
        "coef": coef_simple,
        "se": se_simple,
        "p": p_simple,
        "ci": ci_simple,
        "r2": model_simple.rsquared,
        "adj_r2": model_simple.rsquared_adj,
        "std_effect": std_effect_simple,
    },
    "controls": {
        "coef": coef_ctrl,
        "se": se_ctrl,
        "p": p_ctrl,
        "ci": ci_ctrl,
        "r2": model_controls.rsquared,
        "adj_r2": model_controls.rsquared_adj,
        "std_effect": std_effect_ctrl,
    },
}

print(json.dumps(results, indent=2))
