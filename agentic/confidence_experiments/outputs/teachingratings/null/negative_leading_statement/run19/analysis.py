import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic correlation
pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
spearman_r, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# With controls
# Convert categorical variables using C() in formula
controls = "C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + age + students + allstudents"
model_controls = smf.ols(f"eval ~ beauty + {controls}", data=df).fit(cov_type="HC3")

# Collect results
results = {
    "n": int(df.shape[0]),
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "simple_coef": float(model_simple.params.get("beauty", np.nan)),
    "simple_p": float(model_simple.pvalues.get("beauty", np.nan)),
    "simple_ci": [float(x) for x in model_simple.conf_int().loc["beauty"].values],
    "controls_coef": float(model_controls.params.get("beauty", np.nan)),
    "controls_p": float(model_controls.pvalues.get("beauty", np.nan)),
    "controls_ci": [float(x) for x in model_controls.conf_int().loc["beauty"].values],
}

print(json.dumps(results, indent=2))
