import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ["masfem", "wind", "min", "category", "alldeaths", "year"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create log-transformed deaths to handle skew
# Add 1 to handle zero deaths

df["log_deaths"] = np.log1p(df["alldeaths"])

# Primary model: log deaths on femininity + storm intensity controls
# Controls: wind speed, minimum pressure, category, year (to capture time trends)
model = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=df).fit()

# Also check binary gender as robustness
model_gender = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=df).fit()

print("=== OLS: log(deaths+1) ~ masfem + controls ===")
print(model.summary())
print("\n=== OLS: log(deaths+1) ~ gender_mf + controls ===")
print(model_gender.summary())

# Simple correlation for context
corr = df[["masfem", "alldeaths", "log_deaths"]].corr()
print("\n=== Correlations ===")
print(corr)
