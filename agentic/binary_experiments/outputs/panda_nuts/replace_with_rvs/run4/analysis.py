import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("panda_nuts.csv")

# Define efficiency as nuts opened per second
_df["efficiency"] = _df["nuts_opened"] / _df["seconds"]

# Fit linear model with age, sex, and help as predictors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=_df).fit()

# Summaries for interpretation
print("Model summary:\n", model.summary())
print("\nGroup means (efficiency):")
print(_df.groupby(["sex", "help"])['efficiency'].mean())
