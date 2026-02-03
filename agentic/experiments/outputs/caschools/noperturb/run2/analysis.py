import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "caschools.csv"
df = pd.read_csv(path)

# Compute student-teacher ratio
# Avoid division issues if any teachers are zero (shouldn't be, but guard anyway)
df = df.copy()
df["str"] = df["students"] / df["teachers"]

# Academic performance metrics
# Use reading, math, and their average

df["avg_score"] = (df["read"] + df["math"]) / 2.0

# Basic correlation
corr_read = df["str"].corr(df["read"])
corr_math = df["str"].corr(df["math"])
corr_avg = df["str"].corr(df["avg_score"])

# Simple OLS: score ~ STR

def ols_simple(y):
    X = sm.add_constant(df[["str"]])
    model = sm.OLS(df[y], X).fit()
    return model

model_read = ols_simple("read")
model_math = ols_simple("math")
model_avg = ols_simple("avg_score")

# OLS with basic controls commonly associated with performance
controls = ["income", "english", "lunch", "calworks", "expenditure"]
Xc = sm.add_constant(df[["str"] + controls])
model_read_c = sm.OLS(df["read"], Xc).fit()
model_math_c = sm.OLS(df["math"], Xc).fit()
model_avg_c = sm.OLS(df["avg_score"], Xc).fit()

# Summarize key results
summary = {
    "corr_read": corr_read,
    "corr_math": corr_math,
    "corr_avg": corr_avg,
    "simple_read_coef": model_read.params["str"],
    "simple_read_p": model_read.pvalues["str"],
    "simple_math_coef": model_math.params["str"],
    "simple_math_p": model_math.pvalues["str"],
    "simple_avg_coef": model_avg.params["str"],
    "simple_avg_p": model_avg.pvalues["str"],
    "ctrl_read_coef": model_read_c.params["str"],
    "ctrl_read_p": model_read_c.pvalues["str"],
    "ctrl_math_coef": model_math_c.params["str"],
    "ctrl_math_p": model_math_c.pvalues["str"],
    "ctrl_avg_coef": model_avg_c.params["str"],
    "ctrl_avg_p": model_avg_c.pvalues["str"],
}

print("Student-teacher ratio (STR) summary:")
print(df["str"].describe())
print("\nCorrelations:")
for k in ["corr_read", "corr_math", "corr_avg"]:
    print(f"  {k}: {summary[k]:.4f}")

print("\nSimple OLS (score ~ STR):")
print(f"  read coef={summary['simple_read_coef']:.4f}, p={summary['simple_read_p']:.4g}")
print(f"  math coef={summary['simple_math_coef']:.4f}, p={summary['simple_math_p']:.4g}")
print(f"  avg  coef={summary['simple_avg_coef']:.4f}, p={summary['simple_avg_p']:.4g}")

print("\nControlled OLS (score ~ STR + income + english + lunch + calworks + expenditure):")
print(f"  read coef={summary['ctrl_read_coef']:.4f}, p={summary['ctrl_read_p']:.4g}")
print(f"  math coef={summary['ctrl_math_coef']:.4f}, p={summary['ctrl_math_p']:.4g}")
print(f"  avg  coef={summary['ctrl_avg_coef']:.4f}, p={summary['ctrl_avg_p']:.4g}")

# Save summary to CSV for traceability
pd.DataFrame([summary]).to_csv("analysis_summary.csv", index=False)
