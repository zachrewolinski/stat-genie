import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Map columns for clarity
# feature6: beauty rating (mean-centered)
# feature7: overall teaching evaluation score

# Basic model: rating ~ beauty
model_simple = smf.ols("feature7 ~ feature6", data=df).fit(cov_type="HC1")

# Multivariate model with controls
# Categorical controls: feature2, feature4, feature5, feature8, feature9, feature10
# Numeric controls: feature3 (age), feature11 (students participated), feature12 (students enrolled)
formula = (
    "feature7 ~ feature6 + feature3 + feature11 + feature12 "
    "+ C(feature2) + C(feature4) + C(feature5) + C(feature8) + C(feature9) + C(feature10)"
)
model_controls = smf.ols(formula, data=df).fit(cov_type="HC1")

# Collect key results
results = {
    "simple_coef": model_simple.params.get("feature6", float("nan")),
    "simple_p": model_simple.pvalues.get("feature6", float("nan")),
    "controls_coef": model_controls.params.get("feature6", float("nan")),
    "controls_p": model_controls.pvalues.get("feature6", float("nan")),
}

# Save a concise text summary for inspection if needed
with open("analysis_summary.txt", "w") as f:
    f.write("Simple model (HC1):\n")
    f.write(model_simple.summary().as_text())
    f.write("\n\nControlled model (HC1):\n")
    f.write(model_controls.summary().as_text())
    f.write("\n\nKey results:\n")
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

# Print key results to stdout
print(results)
