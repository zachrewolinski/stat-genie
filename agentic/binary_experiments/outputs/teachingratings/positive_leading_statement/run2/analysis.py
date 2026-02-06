import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("teachingratings.csv")

# Basic correlation
corr = _df["beauty"].corr(_df["eval"])

# OLS with controls similar to literature
# Encode categorical variables using patsy
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) "
    "+ C(division) + C(credits) + students + allstudents"
)
model = smf.ols(formula, data=_df).fit()

# Also check robustness with log students/allstudents to reduce scale effects
_df["log_students"] = (_df["students"]).clip(lower=1).apply(lambda x: np.log(x))
_df["log_allstudents"] = (_df["allstudents"]).clip(lower=1).apply(lambda x: np.log(x))
model_log = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + log_students + log_allstudents",
    data=_df,
).fit()

# Print key results for inspection
print("Correlation (beauty, eval):", corr)
print("OLS beauty coef:", model.params.get("beauty"), "p=", model.pvalues.get("beauty"))
print("OLS (log size) beauty coef:", model_log.params.get("beauty"), "p=", model_log.pvalues.get("beauty"))
