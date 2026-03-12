import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Column mapping based on value ranges and info.json descriptions
col_year = "wind"          # year hurricane occurred (1950-2012)
col_deaths = "name"        # total deaths
col_fem = "category"       # coder masculinity-femininity rating (1-11)
col_fem_mturk = "ind"       # MTurk masculinity-femininity rating (1-11)
col_fem_binary = "masfem_mturk"  # binary female indicator
col_cat = "gender_mf"       # Saffir-Simpson category (1-5)
col_wind = "year"           # max wind speed at landfall (mph)
col_pressure = "ndam15"     # minimum pressure at landfall (mb)

# Basic sanity checks
print("rows", len(df))
print("missing deaths", df[col_deaths].isna().sum())
print("missing fem", df[col_fem].isna().sum())

# Correlation between feminine ratings and binary indicator
corr_fem_binary = stats.spearmanr(df[col_fem], df[col_fem_binary])
corr_fem_mturk = stats.spearmanr(df[col_fem], df[col_fem_mturk])
print("spearman fem vs binary", corr_fem_binary)
print("spearman fem vs mturk", corr_fem_mturk)

# Outcome: log(1 + deaths)
analysis_df = df[[col_deaths, col_fem, col_fem_mturk, col_fem_binary, col_cat, col_wind, col_pressure, col_year]].dropna()
analysis_df = analysis_df.copy()
analysis_df["log_deaths"] = np.log1p(analysis_df[col_deaths])


def fit_model(fem_col):
    X = analysis_df[[fem_col, col_cat, col_wind, col_pressure, col_year]]
    X = sm.add_constant(X)
    y = analysis_df["log_deaths"]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model

model_fem = fit_model(col_fem)
model_fem_mturk = fit_model(col_fem_mturk)
model_fem_binary = fit_model(col_fem_binary)

print("\nModel with fem rating (category column):")
print(model_fem.summary())
print("\nModel with fem mturk (ind column):")
print(model_fem_mturk.summary())
print("\nModel with fem binary:")
print(model_fem_binary.summary())

# Unadjusted correlation with deaths
corr_deaths_fem = stats.spearmanr(analysis_df[col_deaths], analysis_df[col_fem])
print("\nSpearman deaths vs fem rating", corr_deaths_fem)

# Interaction with intensity (optional): fem * category
analysis_df["fem_x_cat"] = analysis_df[col_fem] * analysis_df[col_cat]
X_int = analysis_df[[col_fem, col_cat, "fem_x_cat", col_wind, col_pressure, col_year]]
X_int = sm.add_constant(X_int)
model_int = sm.OLS(analysis_df["log_deaths"], X_int).fit(cov_type="HC3")
print("\nModel with fem x category interaction:")
print(model_int.summary())
