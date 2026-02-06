import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DF_PATH = "hurricane.csv"
df = pd.read_csv(DF_PATH)

# Basic transformations
# log1p to reduce skew and handle zeros
_df = df.copy()
_df["log_deaths"] = np.log1p(_df["feature8"])

# Model 1: femininity index only
X1 = sm.add_constant(_df[["feature4"]])
model1 = sm.OLS(_df["log_deaths"], X1).fit()

# Model 2: femininity index + storm severity controls + year
controls = ["feature4", "feature7", "feature5", "feature13", "feature2"]
X2 = sm.add_constant(_df[controls])
model2 = sm.OLS(_df["log_deaths"], X2).fit()

# Model 3: binary female indicator + controls (robustness)
controls2 = ["feature6", "feature7", "feature5", "feature13", "feature2"]
X3 = sm.add_constant(_df[controls2])
model3 = sm.OLS(_df["log_deaths"], X3).fit()

# Model 4: alternative femininity ratings (feature12) + controls
controls3 = ["feature12", "feature7", "feature5", "feature13", "feature2"]
X4 = sm.add_constant(_df[controls3])
model4 = sm.OLS(_df["log_deaths"], X4).fit()

print("Model 1 (log deaths ~ femininity index):")
print(model1.summary())
print("\nModel 2 (log deaths ~ femininity index + controls):")
print(model2.summary())
print("\nModel 3 (log deaths ~ female indicator + controls):")
print(model3.summary())
print("\nModel 4 (log deaths ~ MTurk femininity + controls):")
print(model4.summary())

# Key results for quick reference
print("\nKey coefficients and p-values:")
print("Model2 feature4 coef/p:", model2.params["feature4"], model2.pvalues["feature4"])
print("Model3 feature6 coef/p:", model3.params["feature6"], model3.pvalues["feature6"])
print("Model4 feature12 coef/p:", model4.params["feature12"], model4.pvalues["feature12"])
