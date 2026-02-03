import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("teachingratings.csv")

# Basic OLS: eval on beauty (bivariate)
model_simple = smf.ols("eval ~ beauty", data=_df).fit()

# Add controls commonly used in the literature
# Use categorical indicators for factors
formula_controls = (
    "eval ~ beauty + age + students + allstudents + "
    "C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)"
)
model_controls = smf.ols(formula_controls, data=_df).fit()

# Save key results to a small CSV for inspection if needed
summary_df = pd.DataFrame(
    {
        "model": ["simple", "controls"],
        "beauty_coef": [model_simple.params.get("beauty"), model_controls.params.get("beauty")],
        "beauty_pvalue": [model_simple.pvalues.get("beauty"), model_controls.pvalues.get("beauty")],
        "beauty_ci_low": [model_simple.conf_int().loc["beauty", 0], model_controls.conf_int().loc["beauty", 0]],
        "beauty_ci_high": [model_simple.conf_int().loc["beauty", 1], model_controls.conf_int().loc["beauty", 1]],
        "n": [int(model_simple.nobs), int(model_controls.nobs)],
        "r2": [model_simple.rsquared, model_controls.rsquared],
    }
)
summary_df.to_csv("analysis_results.csv", index=False)

# Print concise output for human check
print(summary_df)
