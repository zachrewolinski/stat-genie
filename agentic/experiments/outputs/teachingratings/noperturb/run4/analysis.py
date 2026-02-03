import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic checks
# Drop rows with missing values in relevant columns
cols = [
    "eval",
    "beauty",
    "age",
    "gender",
    "minority",
    "native",
    "tenure",
    "credits",
    "division",
    "students",
]
df = df[cols].dropna()

# Fit OLS with categorical controls
formula = (
    "eval ~ beauty + age + students + C(gender) + C(minority) + C(native) "
    "+ C(tenure) + C(credits) + C(division)"
)
model = smf.ols(formula, data=df).fit(cov_type="HC3")

print("N:", int(model.nobs))
print("Beauty coef:", model.params.get("beauty"))
print("Beauty p-value:", model.pvalues.get("beauty"))
print(model.summary())

# Also compute simple correlation
corr = df[["eval", "beauty"]].corr().iloc[0, 1]
print("Correlation eval-beauty:", corr)
