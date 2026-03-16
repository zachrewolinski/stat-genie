import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic cleaning: ensure categorical variables are treated as categories
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# Simple correlation
corr = df["beauty"].corr(df["eval"])

# Simple OLS: eval ~ beauty
model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Adjusted OLS with controls
# Include common controls: age, gender, minority, native, tenure, division, credits, students, allstudents
model_adj = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents",
    data=df,
).fit()

# Standardized effect (beta) for beauty in adjusted model
# standardize variables for standardized coefficient
z = df.copy()
for col in ["eval", "beauty", "age", "students", "allstudents"]:
    z[col] = (z[col] - z[col].mean()) / z[col].std(ddof=0)
model_adj_std = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents",
    data=z,
).fit()

summary = {
    "n": len(df),
    "corr_beauty_eval": corr,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_ci_low": model_simple.conf_int().loc["beauty", 0],
    "simple_ci_high": model_simple.conf_int().loc["beauty", 1],
    "adj_coef": model_adj.params["beauty"],
    "adj_p": model_adj.pvalues["beauty"],
    "adj_ci_low": model_adj.conf_int().loc["beauty", 0],
    "adj_ci_high": model_adj.conf_int().loc["beauty", 1],
    "adj_beta_std": model_adj_std.params["beauty"],
    "adj_r2": model_adj.rsquared,
}

print(summary)
