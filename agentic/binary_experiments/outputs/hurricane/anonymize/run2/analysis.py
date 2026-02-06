import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
cols = {
    "feature4": "masfem_index",
    "feature6": "female_name",
    "feature7": "category",
    "feature5": "min_pressure",
    "feature13": "max_wind",
    "feature8": "deaths",
    "feature9": "damage_2013",
    "feature14": "damage_2015",
    "feature2": "year",
}
df = df.rename(columns=cols)

# Basic cleaning
# Ensure numeric
for c in ["masfem_index", "female_name", "category", "min_pressure", "max_wind", "deaths", "damage_2013", "damage_2015", "year"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Log-transform skewed outcomes/controls
# Use log1p for deaths and damages to handle zeros
for c in ["deaths", "damage_2013", "damage_2015"]:
    df[f"log1p_{c}"] = np.log1p(df[c])

# Core model: fatalities as proxy for precautionary outcomes
# Controls for storm severity and exposure
controls = ["category", "min_pressure", "max_wind", "log1p_damage_2013", "year"]

# Build design matrix for femininity index
model_df = df[["log1p_deaths", "masfem_index"] + controls].dropna()
X = model_df[["masfem_index"] + controls]
X = sm.add_constant(X)
y = model_df["log1p_deaths"]

ols_masfem = sm.OLS(y, X).fit(cov_type="HC3")

# Alternative model using binary female indicator
model_df2 = df[["log1p_deaths", "female_name"] + controls].dropna()
X2 = model_df2[["female_name"] + controls]
X2 = sm.add_constant(X2)
y2 = model_df2["log1p_deaths"]

ols_female = sm.OLS(y2, X2).fit(cov_type="HC3")

# Simple group comparison (mean deaths) for context
means = df.groupby("female_name")["deaths"].mean()

# Output key results
print("N used (masfem):", int(model_df.shape[0]))
print(ols_masfem.summary().tables[1])
print("\nN used (female):", int(model_df2.shape[0]))
print(ols_female.summary().tables[1])

print("\nMean deaths by female_name (0=male,1=female):")
print(means)
