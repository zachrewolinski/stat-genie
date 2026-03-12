import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic checks
n = len(df)

# Pearson correlation between beauty and eval
pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])

# Spearman correlation
spearman_r, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Multiple regression with controls
# Use common controls available in dataset
formula_controls = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents"
)
model_controls = smf.ols(formula_controls, data=df).fit(cov_type="HC3")

# Save summary stats to a dict for easy access
results = {
    "n": n,
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "spearman_r": spearman_r,
    "spearman_p": spearman_p,
    "simple_coef": model_simple.params["beauty"],
    "simple_se": model_simple.bse["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_ci_low": model_simple.conf_int().loc["beauty", 0],
    "simple_ci_high": model_simple.conf_int().loc["beauty", 1],
    "controls_coef": model_controls.params["beauty"],
    "controls_se": model_controls.bse["beauty"],
    "controls_p": model_controls.pvalues["beauty"],
    "controls_ci_low": model_controls.conf_int().loc["beauty", 0],
    "controls_ci_high": model_controls.conf_int().loc["beauty", 1],
    "r2_simple": model_simple.rsquared,
    "r2_controls": model_controls.rsquared,
}

# Print results in a readable way
print("N:", results["n"])
print("Pearson r:", results["pearson_r"], "p=", results["pearson_p"])
print("Spearman r:", results["spearman_r"], "p=", results["spearman_p"])
print("Simple OLS coef (beauty):", results["simple_coef"], "SE=", results["simple_se"], "p=", results["simple_p"], "CI=", (results["simple_ci_low"], results["simple_ci_high"]))
print("Controls OLS coef (beauty):", results["controls_coef"], "SE=", results["controls_se"], "p=", results["controls_p"], "CI=", (results["controls_ci_low"], results["controls_ci_high"]))
print("R2 simple:", results["r2_simple"], "R2 controls:", results["r2_controls"])
