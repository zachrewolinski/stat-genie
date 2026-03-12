import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Rename for clarity
rename_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}
df = df.rename(columns=rename_map)

# Compute efficiency as nuts opened per minute
# Add a small epsilon to avoid division by zero (though min duration is > 0)
eps = 1e-9
df["efficiency"] = df["nuts_opened"] / (df["duration_sec"] + eps) * 60.0

# Ensure categorical types
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# OLS model: efficiency ~ age + sex + help
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Log1p efficiency model for robustness
# (helps if efficiency is skewed)
df["log_efficiency"] = np.log1p(df["efficiency"])
log_model = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Collect key statistics
results = {
    "n": int(model.nobs),
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
    "f_pvalue": float(model.f_pvalue),
}

predictors = ["age", "C(sex)[T.m]", "C(help)[T.y]"]

coef_info = {}
for pred in predictors:
    if pred in model.params.index:
        coef_info[pred] = {
            "coef": float(model.params[pred]),
            "std_err": float(model.bse[pred]),
            "t": float(model.tvalues[pred]),
            "p": float(model.pvalues[pred]),
        }
    else:
        coef_info[pred] = None

log_coef_info = {}
for pred in predictors:
    if pred in log_model.params.index:
        log_coef_info[pred] = {
            "coef": float(log_model.params[pred]),
            "std_err": float(log_model.bse[pred]),
            "t": float(log_model.tvalues[pred]),
            "p": float(log_model.pvalues[pred]),
        }
    else:
        log_coef_info[pred] = None

# Output concise results
print("OLS efficiency model (HC3):")
print(model.summary())
print("\nLog1p efficiency model (HC3):")
print(log_model.summary())
print("\nModel stats:", results)
print("\nCoefficients:", coef_info)
print("\nLog coefficients:", log_coef_info)
