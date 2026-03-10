import json
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Map columns to semantic names from info.json
rename_map = {
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "rating",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure",
    "feature11": "n_eval",
    "feature12": "n_enroll",
    "feature13": "instructor_id",
}

df = df.rename(columns=rename_map)

# Basic stats
beauty_mean = df["beauty"].mean()
beauty_std = df["beauty"].std()
rating_mean = df["rating"].mean()
rating_std = df["rating"].std()

corr = df[["beauty", "rating"]].corr().iloc[0, 1]

# Simple bivariate regression
model_simple = smf.ols("rating ~ beauty", data=df).fit()

# Multivariate regression with common controls
formula = (
    "rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) "
    "+ C(division) + C(native_english) + C(tenure) + n_eval + n_enroll"
)
model_controls = smf.ols(formula, data=df).fit()

# Compute standardized effect of beauty in controlled model
# One SD increase in beauty -> coefficient * SD(beauty)
beauty_coef = model_controls.params["beauty"]
beauty_p = model_controls.pvalues["beauty"]
beauty_effect_sd = beauty_coef * beauty_std

# Get R-squared values
r2_simple = model_simple.rsquared
r2_controls = model_controls.rsquared

results = {
    "n": int(df.shape[0]),
    "beauty_mean": beauty_mean,
    "beauty_std": beauty_std,
    "rating_mean": rating_mean,
    "rating_std": rating_std,
    "corr_beauty_rating": corr,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_r2": r2_simple,
    "controls_coef": beauty_coef,
    "controls_p": beauty_p,
    "controls_r2": r2_controls,
    "beauty_effect_sd_in_rating": beauty_effect_sd,
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
