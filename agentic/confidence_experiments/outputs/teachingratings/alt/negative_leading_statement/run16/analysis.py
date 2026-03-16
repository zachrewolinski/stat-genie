import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning and derived variables
# Participation rate: fraction of enrolled students who completed evaluations

df["participation"] = df["students"] / df["allstudents"]

# Ensure categorical variables are treated as categories
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Model 1: simple relationship
model1 = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Model 2: adjusted with controls
model2 = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + allstudents + participation",
    data=df,
).fit(cov_type="HC3")

# Standardized effect of beauty (per 1 SD increase in beauty)
beauty_sd = df["beauty"].std()

# Extract key results
results = {
    "n": int(df.shape[0]),
    "beauty_sd": beauty_sd,
    "model1": {
        "coef": model1.params["beauty"],
        "se": model1.bse["beauty"],
        "p": model1.pvalues["beauty"],
        "ci_low": model1.conf_int().loc["beauty", 0],
        "ci_high": model1.conf_int().loc["beauty", 1],
        "r2": model1.rsquared,
    },
    "model2": {
        "coef": model2.params["beauty"],
        "se": model2.bse["beauty"],
        "p": model2.pvalues["beauty"],
        "ci_low": model2.conf_int().loc["beauty", 0],
        "ci_high": model2.conf_int().loc["beauty", 1],
        "r2": model2.rsquared,
    },
}

# Effect per 1 SD increase in beauty
results["model1"]["effect_per_sd"] = results["model1"]["coef"] * beauty_sd
results["model2"]["effect_per_sd"] = results["model2"]["coef"] * beauty_sd

# Save results for inspection
pd.Series(results).to_json("analysis_results.json")

print(results)
