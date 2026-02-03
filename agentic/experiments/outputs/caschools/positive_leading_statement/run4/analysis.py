import pandas as pd
import statsmodels.api as sm

DATA_PATH = "caschools.csv"

df = pd.read_csv(DATA_PATH)

# Student-teacher ratio
if "students" not in df.columns or "teachers" not in df.columns:
    raise ValueError("Expected columns 'students' and 'teachers' in dataset.")

df["str"] = df["students"] / df["teachers"]

# Academic performance: average of reading and math scores
for col in ["read", "math"]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in dataset.")

df["performance"] = df[["read", "math"]].mean(axis=1)

# Simple correlation
corr = df["str"].corr(df["performance"])

# Simple OLS: performance ~ str
X_simple = sm.add_constant(df[["str"]])
model_simple = sm.OLS(df["performance"], X_simple).fit()

# Controlled OLS with key covariates
controls = ["lunch", "income", "english", "expenditure"]
controls = [c for c in controls if c in df.columns]
X_ctrl = sm.add_constant(df[["str"] + controls])
model_ctrl = sm.OLS(df["performance"], X_ctrl).fit()

# Save a brief report for inspection
report = {
    "corr_str_performance": corr,
    "simple_coef_str": model_simple.params.get("str"),
    "simple_p_str": model_simple.pvalues.get("str"),
    "ctrl_coef_str": model_ctrl.params.get("str"),
    "ctrl_p_str": model_ctrl.pvalues.get("str"),
    "n": int(df.shape[0]),
}

print(report)
