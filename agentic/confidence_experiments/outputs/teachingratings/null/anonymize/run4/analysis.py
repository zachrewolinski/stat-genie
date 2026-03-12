import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Map features to names for clarity
cols = {
    "feature1": "course_id",
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "evaluation",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure_track",
    "feature11": "students_eval",
    "feature12": "students_enrolled",
    "feature13": "instructor_id",
}

_df = _df.rename(columns=cols)

# Basic stats
n = len(_df)

# Correlation between beauty and evaluation
corr = _df["beauty"].corr(_df["evaluation"])

# Simple OLS
simple_model = smf.ols("evaluation ~ beauty", data=_df).fit(cov_type="HC3")

# Controls: demographics + course characteristics + class size
# Encode categorical factors with treatment coding via formula
formula = (
    "evaluation ~ beauty + age + C(gender) + C(minority) + C(single_credit) + "
    "C(division) + C(native_english) + C(tenure_track) + students_enrolled + students_eval"
)

full_model = smf.ols(formula, data=_df).fit(cov_type="HC3")

# Standardized effect: beauty SD -> evaluation points
beauty_sd = _df["beauty"].std(ddof=1)
coef = full_model.params.get("beauty", np.nan)
se = full_model.bse.get("beauty", np.nan)
pval = full_model.pvalues.get("beauty", np.nan)
ci_low, ci_high = full_model.conf_int().loc["beauty"].tolist()

std_effect = coef * beauty_sd

# R-squared from full model
r2 = full_model.rsquared

# Build summary
summary = {
    "n": n,
    "corr_beauty_eval": corr,
    "simple_coef": simple_model.params.get("beauty", np.nan),
    "simple_pval": simple_model.pvalues.get("beauty", np.nan),
    "full_coef": coef,
    "full_se": se,
    "full_pval": pval,
    "full_ci": [ci_low, ci_high],
    "beauty_sd": beauty_sd,
    "std_effect_eval_points_per_1sd_beauty": std_effect,
    "full_r2": r2,
}

print(json.dumps(summary, indent=2))
