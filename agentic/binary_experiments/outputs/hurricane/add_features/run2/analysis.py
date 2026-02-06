import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("hurricane.csv")

# Keep relevant columns and drop rows with missing values
cols = [
    "alldeaths",
    "masfem",
    "gender_mf",
    "wind",
    "min",
    "category",
    "ndam15",
    "year",
    "elapsedyrs",
]

df = _df[cols].copy()

df = df.dropna()

# Outcome: log(1+deaths) to reduce skew
# Add small constant to avoid log(0)
df["log_deaths"] = np.log1p(df["alldeaths"])

# Standardize continuous predictors for interpretability
for c in ["masfem", "wind", "min", "ndam15", "year", "elapsedyrs"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

# Model 1: masfem + hurricane intensity controls
X1 = df[["masfem_z", "wind_z", "min_z", "category", "ndam15_z"]]
X1 = sm.add_constant(X1)
model1 = sm.OLS(df["log_deaths"], X1).fit(cov_type="HC3")

# Model 2: binary gender + same controls
X2 = df[["gender_mf", "wind_z", "min_z", "category", "ndam15_z"]]
X2 = sm.add_constant(X2)
model2 = sm.OLS(df["log_deaths"], X2).fit(cov_type="HC3")

# Model 3: add year to account for time trends
X3 = df[["masfem_z", "wind_z", "min_z", "category", "ndam15_z", "year_z"]]
X3 = sm.add_constant(X3)
model3 = sm.OLS(df["log_deaths"], X3).fit(cov_type="HC3")

# Print concise results
print("N:", len(df))

def summarize(model, key):
    coef = model.params[key]
    se = model.bse[key]
    p = model.pvalues[key]
    return coef, se, p

print("Model1 (masfem):", summarize(model1, "masfem_z"))
print("Model2 (gender_mf):", summarize(model2, "gender_mf"))
print("Model3 (masfem + year):", summarize(model3, "masfem_z"))
