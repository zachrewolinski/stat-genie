import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic info
n_rows = len(df)

# Ensure numeric columns
for col in ["beauty", "eval", "age", "students", "allstudents"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing key variables
key_df = df.dropna(subset=["beauty", "eval"])

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(key_df["beauty"], key_df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=key_df).fit()

# Multivariate OLS with common controls from teaching ratings dataset
# Use only columns that exist and are non-null enough
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(division) + "
    "C(native) + C(tenure) + C(credits) + students + allstudents"
)

# Some columns may have missing values; drop rows with NA in model variables
model_df = key_df[[
    "eval", "beauty", "age", "gender", "minority", "division", "native",
    "tenure", "credits", "students", "allstudents"
]].dropna()

model_multi = smf.ols(formula, data=model_df).fit()

# Compute standardized effect for beauty in simple and multivariate models
# Standardize by SD of eval and beauty in the respective data
beauty_sd_simple = key_df["beauty"].std()
beauty_sd_multi = model_df["beauty"].std()

eval_sd_simple = key_df["eval"].std()
eval_sd_multi = model_df["eval"].std()

std_beta_simple = model_simple.params["beauty"] * (beauty_sd_simple / eval_sd_simple)
std_beta_multi = model_multi.params["beauty"] * (beauty_sd_multi / eval_sd_multi)

# Partial R^2 for beauty in multivariate model
# partial R^2 = t^2 / (t^2 + df_resid)
beauty_t = model_multi.tvalues["beauty"]
partial_r2 = (beauty_t ** 2) / (beauty_t ** 2 + model_multi.df_resid)

results = {
    "n_rows": int(n_rows),
    "n_used_corr": int(len(key_df)),
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "simple_coef": float(model_simple.params["beauty"]),
    "simple_p": float(model_simple.pvalues["beauty"]),
    "simple_r2": float(model_simple.rsquared),
    "simple_std_beta": float(std_beta_simple),
    "multi_n": int(model_multi.nobs),
    "multi_coef": float(model_multi.params["beauty"]),
    "multi_p": float(model_multi.pvalues["beauty"]),
    "multi_r2": float(model_multi.rsquared),
    "multi_std_beta": float(std_beta_multi),
    "partial_r2": float(partial_r2),
}

print(json.dumps(results, indent=2))
