import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Rename for readability
cols = {
    "feature6": "beauty",
    "feature7": "rating",
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure_track",
    "feature11": "students_evaluated",
    "feature12": "students_enrolled",
}

df = df.rename(columns=cols)

# Basic correlation
corr = df["beauty"].corr(df["rating"])

# Simple OLS
model_simple = smf.ols("rating ~ beauty", data=df).fit(cov_type="HC3")

# Multiple regression with controls
model_controls = smf.ols(
    "rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + C(division) + C(native_english) + C(tenure_track) + students_evaluated + students_enrolled",
    data=df,
).fit(cov_type="HC3")

# Standardized effect size for beauty (per SD of beauty)
beauty_sd = df["beauty"].std()
coef_simple = model_simple.params["beauty"]
coef_controls = model_controls.params["beauty"]

std_effect_simple = coef_simple * beauty_sd
std_effect_controls = coef_controls * beauty_sd

results = {
    "n": int(df.shape[0]),
    "corr_beauty_rating": corr,
    "simple_coef": coef_simple,
    "simple_p": model_simple.pvalues["beauty"],
    "simple_ci": model_simple.conf_int().loc["beauty"].tolist(),
    "controls_coef": coef_controls,
    "controls_p": model_controls.pvalues["beauty"],
    "controls_ci": model_controls.conf_int().loc["beauty"].tolist(),
    "std_effect_simple": std_effect_simple,
    "std_effect_controls": std_effect_controls,
    "r2_simple": model_simple.rsquared,
    "r2_controls": model_controls.rsquared,
}

print(json.dumps(results, indent=2))
