import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic summary
n = len(df)

# Simple correlation and regression
corr = df[["beauty", "eval"]].corr().iloc[0,1]

model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Multiple regression with controls (categorical with C())
formula = (
    "eval ~ beauty + age + students + allstudents + "
    "C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
)
model_controls = smf.ols(formula, data=df).fit()

# Alternative: cluster by prof? (not necessary but note)
# We can compute robust SE (HC3) for sensitivity
model_controls_hc3 = model_controls.get_robustcov_results(cov_type="HC3")

# Collect key stats
results = {
    "n": n,
    "corr": corr,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_r2": model_simple.rsquared,
    "controls_coef": model_controls.params["beauty"],
    "controls_p": model_controls.pvalues["beauty"],
    "controls_r2": model_controls.rsquared,
    "controls_hc3_p": model_controls_hc3.pvalues[model_controls_hc3.model.exog_names.index("beauty")],
}

print(results)
