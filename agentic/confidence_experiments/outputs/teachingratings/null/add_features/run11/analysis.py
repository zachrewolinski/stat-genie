import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing in key variables
key_vars = ["eval", "beauty"]
for col in key_vars:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Ensure numeric types
for col in ["eval", "beauty", "age", "students", "allstudents"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing in variables used for each model
simple_df = df.dropna(subset=["eval", "beauty"]).copy()

# Simple OLS: eval ~ beauty
simple_model = smf.ols("eval ~ beauty", data=simple_df).fit(cov_type="HC3")

# Controlled OLS with common covariates
controls = [
    "age",
    "C(gender)",
    "C(minority)",
    "C(native)",
    "C(tenure)",
    "C(division)",
    "C(credits)",
    "students",
    "allstudents",
]

# Keep only controls that exist in the dataset
existing_controls = []
for term in controls:
    raw = term.replace("C(", "").replace(")", "")
    if raw in df.columns:
        existing_controls.append(term)

control_formula = " + ".join(existing_controls)
full_formula = f"eval ~ beauty" + (f" + {control_formula}" if control_formula else "")

control_df = df.dropna(subset=["eval", "beauty"] + [c.replace("C(", "").replace(")", "") for c in existing_controls]).copy()
control_model = smf.ols(full_formula, data=control_df).fit(cov_type="HC3")

print("N_simple", len(simple_df))
print("N_control", len(control_df))
print("Simple coef", simple_model.params["beauty"], "p", simple_model.pvalues["beauty"])
print("Control coef", control_model.params["beauty"], "p", control_model.pvalues["beauty"])
print("Simple R2", simple_model.rsquared)
print("Control R2", control_model.rsquared)
