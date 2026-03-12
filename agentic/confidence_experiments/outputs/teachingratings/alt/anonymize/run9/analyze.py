import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic correlation
x = df["feature6"]
y = df["feature7"]
pearson_r, pearson_p = stats.pearsonr(x, y)

# Simple OLS
model_simple = smf.ols("feature7 ~ feature6", data=df).fit()

# OLS with controls
formula_controls = (
    "feature7 ~ feature6 + feature3 + feature11 + feature12 + "
    "C(feature2) + C(feature4) + C(feature5) + C(feature8) + C(feature9) + C(feature10)"
)
model_controls = smf.ols(formula_controls, data=df).fit()

# Standardized coefficient for feature6 in control model
# standardize x and y
x_std = (df["feature6"] - df["feature6"].mean()) / df["feature6"].std(ddof=0)
y_std = (df["feature7"] - df["feature7"].mean()) / df["feature7"].std(ddof=0)
model_std = smf.ols("y_std ~ x_std", data={"y_std": y_std, "x_std": x_std}).fit()

results = {
    "n": int(df.shape[0]),
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "simple_coef": model_simple.params["feature6"],
    "simple_p": model_simple.pvalues["feature6"],
    "simple_r2": model_simple.rsquared,
    "controls_coef": model_controls.params["feature6"],
    "controls_p": model_controls.pvalues["feature6"],
    "controls_r2": model_controls.rsquared,
    "std_coef_simple": model_std.params["x_std"],
}

print(json.dumps(results, indent=2))

# Save model summary key lines for checking
summary = {
    "simple_ci": model_simple.conf_int().loc["feature6"].tolist(),
    "controls_ci": model_controls.conf_int().loc["feature6"].tolist(),
}
print("\nCIs:")
print(json.dumps(summary, indent=2))
