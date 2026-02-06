import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic prep
_df["log_deaths"] = np.log1p(_df["alldeaths"].astype(float))

# Select controls commonly used for severity
controls = ["wind", "min", "category"]

# Drop rows with missing values in key columns
cols_masfem = ["log_deaths", "masfem"] + controls
cols_gender = ["log_deaths", "gender_mf"] + controls

_df_m = _df[cols_masfem].dropna()
_df_g = _df[cols_gender].dropna()


def fit_ols(df, y_col, x_cols):
    X = df[x_cols]
    X = sm.add_constant(X)
    y = df[y_col]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


# Model 1: feminity scale
m1 = fit_ols(_df_m, "log_deaths", ["masfem"] + controls)

# Model 2: binary gender
m2 = fit_ols(_df_g, "log_deaths", ["gender_mf"] + controls)

# Additional model: include ndam15 as a severity proxy (log-transformed)
_df["log_ndam15"] = np.log1p(_df["ndam15"].astype(float))
cols_masfem_dam = ["log_deaths", "masfem", "log_ndam15"] + controls
_df_md = _df[cols_masfem_dam].dropna()

m3 = fit_ols(_df_md, "log_deaths", ["masfem", "log_ndam15"] + controls)


# Output key results
print("Rows total:", len(_df))
print("Rows used (masfem):", len(_df_m))
print("Rows used (gender_mf):", len(_df_g))
print("Rows used (masfem + damages):", len(_df_md))

print("\nModel 1: log_deaths ~ masfem + wind + min + category")
print(m1.summary().tables[1])

print("\nModel 2: log_deaths ~ gender_mf + wind + min + category")
print(m2.summary().tables[1])

print("\nModel 3: log_deaths ~ masfem + log_ndam15 + wind + min + category")
print(m3.summary().tables[1])

# Save a compact results CSV for record
out = pd.DataFrame([
    {
        "model": "m1",
        "coef": m1.params.get("masfem", np.nan),
        "pvalue": m1.pvalues.get("masfem", np.nan),
    },
    {
        "model": "m2",
        "coef": m2.params.get("gender_mf", np.nan),
        "pvalue": m2.pvalues.get("gender_mf", np.nan),
    },
    {
        "model": "m3",
        "coef": m3.params.get("masfem", np.nan),
        "pvalue": m3.pvalues.get("masfem", np.nan),
    },
])

out.to_csv("analysis_results.csv", index=False)
