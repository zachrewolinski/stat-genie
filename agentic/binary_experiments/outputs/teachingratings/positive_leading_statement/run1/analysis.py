import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "teachingratings.csv"
df = pd.read_csv(DF_PATH)

# Basic sanity checks
assert "beauty" in df.columns and "eval" in df.columns

# Simple correlation
corr = df["beauty"].corr(df["eval"])

# OLS without controls
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# OLS with controls (categorical factors + numeric controls)
formula = (
    "eval ~ beauty + age + students + allstudents "
    "+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
)
model_controls = smf.ols(formula, data=df).fit(cov_type="HC3")

results = {
    "corr_beauty_eval": corr,
    "simple_coef": model_simple.params.get("beauty"),
    "simple_p": model_simple.pvalues.get("beauty"),
    "controls_coef": model_controls.params.get("beauty"),
    "controls_p": model_controls.pvalues.get("beauty"),
}

print("Correlation(beauty, eval):", results["corr_beauty_eval"])
print("Simple OLS coef/p:", results["simple_coef"], results["simple_p"])
print("Controls OLS coef/p:", results["controls_coef"], results["controls_p"])
