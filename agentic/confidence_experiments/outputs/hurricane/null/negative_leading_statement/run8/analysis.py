import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("hurricane.csv")

# Basic cleaning
# Drop rows with missing key variables
key_cols = ["alldeaths", "masfem", "wind", "min", "category", "year"]
clean = df.dropna(subset=key_cols).copy()

clean["log_deaths"] = np.log1p(clean["alldeaths"])

results = {}

# Model A: simple association
model_a = smf.ols("log_deaths ~ masfem", data=clean).fit(cov_type="HC3")
results["model_a"] = {
    "coef": model_a.params.get("masfem"),
    "p": model_a.pvalues.get("masfem"),
    "n": int(model_a.nobs),
    "r2": model_a.rsquared,
}

# Model B: control for hurricane severity and time trend
model_b = smf.ols(
    "log_deaths ~ masfem + category + wind + min + year",
    data=clean,
).fit(cov_type="HC3")
results["model_b"] = {
    "coef": model_b.params.get("masfem"),
    "p": model_b.pvalues.get("masfem"),
    "n": int(model_b.nobs),
    "r2": model_b.rsquared,
}

# Model C: binary female indicator
model_c = smf.ols(
    "log_deaths ~ gender_mf + category + wind + min + year",
    data=clean,
).fit(cov_type="HC3")
results["model_c"] = {
    "coef": model_c.params.get("gender_mf"),
    "p": model_c.pvalues.get("gender_mf"),
    "n": int(model_c.nobs),
    "r2": model_c.rsquared,
}

# Negative binomial on counts (if it converges)
try:
    nb = smf.glm(
        "alldeaths ~ masfem + category + wind + min + year",
        data=clean,
        family=sm.families.NegativeBinomial(),
    ).fit()
    results["model_nb"] = {
        "coef": nb.params.get("masfem"),
        "p": nb.pvalues.get("masfem"),
        "n": int(nb.nobs),
        "aic": nb.aic,
    }
except Exception as e:
    results["model_nb"] = {"error": str(e)}

# Save numerical outputs for later inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
