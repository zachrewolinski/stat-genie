import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Outcome: deaths (skewed) -> log1p transform
df["log_deaths"] = np.log1p(df["alldeaths"])

# Core predictors
predictors = ["masfem", "wind", "min", "category", "ndam15"]
X = df[predictors]
X = sm.add_constant(X)

y = df["log_deaths"]

model = sm.OLS(y, X).fit()

# Also compare binary gender indicator (female=1) with same controls
predictors_gender = ["gender_mf", "wind", "min", "category", "ndam15"]
X2 = sm.add_constant(df[predictors_gender])
model_gender = sm.OLS(y, X2).fit()

print("OLS log1p(deaths) ~ masfem + wind + min + category + ndam15")
print(model.summary())
print("\nOLS log1p(deaths) ~ gender_mf + wind + min + category + ndam15")
print(model_gender.summary())

# Simple bivariate correlation for reference
corr = df[["masfem", "alldeaths"]].corr().iloc[0, 1]
print(f"\nPearson correlation masfem vs alldeaths: {corr:.3f}")
