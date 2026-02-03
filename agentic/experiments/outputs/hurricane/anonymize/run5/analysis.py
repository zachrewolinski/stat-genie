import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("hurricane.csv")

# Rename columns for readability
cols = {
    "feature1": "id",
    "feature2": "year",
    "feature3": "name",
    "feature4": "masfem",
    "feature5": "min_pressure",
    "feature6": "female",
    "feature7": "category",
    "feature8": "deaths",
    "feature9": "damage2013",
    "feature10": "years_elapsed",
    "feature11": "source",
    "feature12": "masfem_mturk",
    "feature13": "max_wind",
    "feature14": "damage2015",
}
_df = _df.rename(columns=cols)

# Basic checks
_df["log_deaths"] = np.log1p(_df["deaths"])

# Model 1: femininity index only
X1 = sm.add_constant(_df[["masfem"]])
model1 = sm.OLS(_df["log_deaths"], X1).fit()

# Model 2: femininity index + controls for storm intensity and time
X2 = sm.add_constant(_df[["masfem", "category", "min_pressure", "max_wind", "year"]])
model2 = sm.OLS(_df["log_deaths"], X2).fit()

# Model 3: binary female + same controls
X3 = sm.add_constant(_df[["female", "category", "min_pressure", "max_wind", "year"]])
model3 = sm.OLS(_df["log_deaths"], X3).fit()

# Group means for intuition
mean_by_gender = _df.groupby("female")["deaths"].mean()

print("Rows:", len(_df))
print("Mean deaths by gender (0=male,1=female):")
print(mean_by_gender)
print("\nModel 1 (log deaths ~ masfem):")
print(model1.summary())
print("\nModel 2 (log deaths ~ masfem + controls):")
print(model2.summary())
print("\nModel 3 (log deaths ~ female + controls):")
print(model3.summary())
