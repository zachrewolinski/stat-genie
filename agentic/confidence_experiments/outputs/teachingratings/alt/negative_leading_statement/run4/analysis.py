import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic clean
# Ensure categorical columns are treated as category
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

# Simple correlation
corr = df["beauty"].corr(df["eval"])

# Simple OLS: eval ~ beauty
model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Multiple regression with controls
# Choose reasonable controls: age, gender, minority, division, credits, native, tenure, students, allstudents
# Avoid multicollinearity: students and allstudents highly related; include students only for parsimony.
controls = "age + C(gender) + C(minority) + C(division) + C(credits) + C(native) + C(tenure) + students"
formula = f"eval ~ beauty + {controls}"
model_controls = smf.ols(formula, data=df).fit()

# Robust SE (HC3)
model_controls_hc3 = smf.ols(formula, data=df).fit(cov_type="HC3")
model_simple_hc3 = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Extract key stats
results = {
    "corr": corr,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_r2": model_simple.rsquared,
    "simple_hc3_p": model_simple_hc3.pvalues["beauty"],
    "controls_coef": model_controls.params["beauty"],
    "controls_p": model_controls.pvalues["beauty"],
    "controls_r2": model_controls.rsquared,
    "controls_hc3_p": model_controls_hc3.pvalues["beauty"],
}

print("RESULTS")
for k, v in results.items():
    print(f"{k}: {v}")

# Also compute standardized effect size for beauty in simple model (beta coefficient)
# Standardized beta = coef * (std_x / std_y)
std_x = df["beauty"].std()
std_y = df["eval"].std()
std_beta = model_simple.params["beauty"] * (std_x / std_y)
print("std_beta_simple:", std_beta)

# Compute effect size: predicted difference between +1 SD and -1 SD beauty
pred_diff = model_simple.params["beauty"] * (2 * std_x)
print("pred_diff_eval_simple_2sd:", pred_diff)

# Save for manual reasoning if needed
