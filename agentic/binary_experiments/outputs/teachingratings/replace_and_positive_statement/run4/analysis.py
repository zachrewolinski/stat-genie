import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("teachingratings.csv")

# Basic correlation between beauty and eval
corr = df["beauty"].corr(df["eval"])

# Simple regression
model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Multiple regression with controls
model_controls = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents",
    data=df,
).fit(cov_type="HC3")

# Print key results
print("Correlation beauty-eval:", corr)
print("\nSimple regression (eval ~ beauty):")
print(model_simple.summary().tables[1])
print("\nControls regression (robust SE):")
print(model_controls.summary().tables[1])

# Extract stats for conclusion
beauty_coef = model_controls.params["beauty"]
beauty_p = model_controls.pvalues["beauty"]
print("\nBeauty coef (controls):", beauty_coef)
print("Beauty p-value (controls):", beauty_p)
