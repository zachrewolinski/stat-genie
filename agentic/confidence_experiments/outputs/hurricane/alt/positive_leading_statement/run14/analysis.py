import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Prepare variables
# log1p for deaths (handles zeros)
df["log_deaths"] = np.log1p(df["alldeaths"])
# log1p for damage (normalized)
df["log_ndam15"] = np.log1p(df["ndam15"])

# Define predictors
predictors_basic = ["masfem"]
predictors_controls = ["masfem", "wind", "min", "category", "log_ndam15"]

# Drop rows with missing in any used columns
cols_basic = ["log_deaths"] + predictors_basic
cols_controls = ["log_deaths"] + predictors_controls

df_basic = df[cols_basic].dropna()
df_controls = df[cols_controls].dropna()

# Correlations
pearson_r, pearson_p = stats.pearsonr(df_basic["masfem"], df_basic["log_deaths"])
spearman_r, spearman_p = stats.spearmanr(df_basic["masfem"], df_basic["log_deaths"])

# OLS models
X_basic = sm.add_constant(df_basic[predictors_basic])
model_basic = sm.OLS(df_basic["log_deaths"], X_basic).fit(cov_type="HC3")

X_controls = sm.add_constant(df_controls[predictors_controls])
model_controls = sm.OLS(df_controls["log_deaths"], X_controls).fit(cov_type="HC3")

# Also test binary gender indicator
cols_gender = ["log_deaths", "gender_mf", "wind", "min", "category", "log_ndam15"]
df_gender = df[cols_gender].dropna()
X_gender = sm.add_constant(df_gender[["gender_mf", "wind", "min", "category", "log_ndam15"]])
model_gender = sm.OLS(df_gender["log_deaths"], X_gender).fit(cov_type="HC3")

results = {
    "n_total": int(len(df)),
    "n_basic": int(len(df_basic)),
    "n_controls": int(len(df_controls)),
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "basic_coef_masfem": float(model_basic.params["masfem"]),
    "basic_p_masfem": float(model_basic.pvalues["masfem"]),
    "controls_coef_masfem": float(model_controls.params["masfem"]),
    "controls_p_masfem": float(model_controls.pvalues["masfem"]),
    "controls_r2": float(model_controls.rsquared),
    "gender_coef": float(model_gender.params["gender_mf"]),
    "gender_p": float(model_gender.pvalues["gender_mf"]),
    "gender_r2": float(model_gender.rsquared),
}

print(json.dumps(results, indent=2))
