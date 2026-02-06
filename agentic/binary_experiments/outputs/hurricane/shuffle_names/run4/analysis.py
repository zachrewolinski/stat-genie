import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "hurricane.csv"
df = pd.read_csv(DATA_PATH)

# Map columns based on observed distributions
# wind: year (1950-2012)
# alldeaths: hurricane name (string)
# category: femininity rating (1-11)
# ndam15: minimum pressure at landfall
# masfem_mturk: binary female indicator (0 male, 1 female)
# gender_mf: Saffir-Simpson category (1-5)
# name: total deaths
# elapsedyrs: normalized damage (missing for some)
# year: wind speed at landfall (mph)

df = df.copy()

# Basic sanity checks
# Remove rows with missing key variables
model_vars = ["name", "category", "masfem_mturk", "gender_mf", "ndam15", "year", "elapsedyrs", "wind"]

# log1p deaths to reduce skew
if (df["name"] < 0).any():
    raise ValueError("Deaths column contains negative values.")

df["log_deaths"] = np.log1p(df["name"])

# Create a cleaned dataset for regression
reg_df = df[model_vars + ["log_deaths"]].dropna()

# Model 1: femininity rating (continuous)
X1 = reg_df[["category", "gender_mf", "ndam15", "year", "elapsedyrs", "wind"]].copy()
X1 = sm.add_constant(X1)
model1 = sm.OLS(reg_df["log_deaths"], X1).fit(cov_type="HC3")

# Model 2: binary female indicator
X2 = reg_df[["masfem_mturk", "gender_mf", "ndam15", "year", "elapsedyrs", "wind"]].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(reg_df["log_deaths"], X2).fit(cov_type="HC3")

# Simple group comparison
female_mean = df.loc[df["masfem_mturk"] == 1, "name"].mean()
male_mean = df.loc[df["masfem_mturk"] == 0, "name"].mean()

print("Rows total:", len(df))
print("Rows used in regression:", len(reg_df))
print("Mean deaths (female names):", round(female_mean, 2))
print("Mean deaths (male names):", round(male_mean, 2))

print("\nModel 1: log(deaths) ~ femininity rating + controls")
print(model1.summary().tables[1])

print("\nModel 2: log(deaths) ~ female indicator + controls")
print(model2.summary().tables[1])

# Save key results for quick review
results = {
    "n_total": int(len(df)),
    "n_reg": int(len(reg_df)),
    "female_mean_deaths": float(female_mean),
    "male_mean_deaths": float(male_mean),
    "model1_fem_coef": float(model1.params["category"]),
    "model1_fem_p": float(model1.pvalues["category"]),
    "model2_fem_coef": float(model2.params["masfem_mturk"]),
    "model2_fem_p": float(model2.pvalues["masfem_mturk"]),
}

pd.Series(results).to_csv("analysis_results.csv")
