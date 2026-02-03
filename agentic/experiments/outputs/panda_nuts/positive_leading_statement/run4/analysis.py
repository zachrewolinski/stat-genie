import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Nut-cracking efficiency: nuts opened per second
_df["efficiency"] = _df["nuts_opened"] / _df["seconds"]

# Basic descriptive stats
print("Rows:", len(_df))
print(_df[["age", "sex", "help", "nuts_opened", "seconds", "efficiency"]].describe(include="all"))

# OLS model with categorical predictors for sex and help
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=_df).fit()
print(model.summary())

# Also show group means for intuition
print("\nGroup means (efficiency):")
print(_df.groupby(["sex", "help"])['efficiency'].mean())
print("\nAge correlation with efficiency:", _df["age"].corr(_df["efficiency"]))
