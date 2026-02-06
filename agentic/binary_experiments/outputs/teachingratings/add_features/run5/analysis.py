import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("teachingratings.csv")

# Select relevant columns
cols = [
    "eval",
    "beauty",
    "age",
    "gender",
    "minority",
    "credits",
    "division",
    "native",
    "tenure",
    "students",
]

# Ensure columns exist
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing values in selected columns
model_df = df[cols].dropna().copy()

# Basic correlation
corr = model_df["eval"].corr(model_df["beauty"])

# Simple regression
model_simple = smf.ols("eval ~ beauty", data=model_df).fit()

# Regression with controls
model_controls = smf.ols(
    "eval ~ beauty + age + students + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)",
    data=model_df,
).fit()

# Extract beauty effects
simple_coef = model_simple.params["beauty"]
simple_p = model_simple.pvalues["beauty"]

control_coef = model_controls.params["beauty"]
control_p = model_controls.pvalues["beauty"]

# Output key results
print("Rows used:", len(model_df))
print("Correlation (eval, beauty):", corr)
print("Simple model: coef=%.4f p=%.4f" % (simple_coef, simple_p))
print("Controls model: coef=%.4f p=%.4f" % (control_coef, control_p))
