import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
_df = pd.read_csv("teachingratings.csv")

# Core variables for the research question
cols = [
    "eval",
    "beauty",
    "age",
    "students",
    "allstudents",
    "gender",
    "minority",
    "native",
    "tenure",
    "division",
    "credits",
]

df = _df[cols].dropna()

# Regression: teaching evaluations on beauty + standard controls
formula = (
    "eval ~ beauty + age + students + allstudents + "
    "C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)"
)
model = smf.ols(formula, data=df).fit()

beauty_coef = model.params["beauty"]
beauty_p = model.pvalues["beauty"]

print("Rows used:", len(df))
print("Beauty coefficient:", round(float(beauty_coef), 4))
print("Beauty p-value:", float(beauty_p))
print("R-squared:", round(float(model.rsquared), 4))

# Simple bivariate correlation for context
corr = df[["eval", "beauty"]].corr().iloc[0, 1]
print("Eval-Beauty correlation:", round(float(corr), 4))
