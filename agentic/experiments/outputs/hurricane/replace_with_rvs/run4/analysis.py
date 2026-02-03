import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
_df = pd.read_csv("hurricane.csv")

# Basic sanity checks
assert {'alldeaths', 'masfem', 'wind', 'min', 'category'}.issubset(_df.columns)

# Create log deaths to reduce skew
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Core regression: log deaths on femininity and storm severity controls
X = _df[['masfem', 'wind', 'min', 'category']]
X = sm.add_constant(X)
model = sm.OLS(_df['log_deaths'], X).fit()

# Alternative check using binary gender indicator
X_bin = _df[['gender_mf', 'wind', 'min', 'category']]
X_bin = sm.add_constant(X_bin)
model_bin = sm.OLS(_df['log_deaths'], X_bin).fit()

# Simple correlations for context
corr_masfem = _df['alldeaths'].corr(_df['masfem'])
corr_gender = _df['alldeaths'].corr(_df['gender_mf'])

print("Rows:", len(_df))
print("Correlation (alldeaths, masfem):", corr_masfem)
print("Correlation (alldeaths, gender_mf):", corr_gender)
print("\nOLS with masfem + controls")
print(model.summary())
print("\nOLS with gender_mf + controls")
print(model_bin.summary())
