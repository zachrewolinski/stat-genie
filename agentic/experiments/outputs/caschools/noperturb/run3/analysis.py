import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("caschools.csv")

# Student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of read and math
df["avg_score"] = df[["read", "math"]].mean(axis=1)

# Simple correlation
corr = df["str"].corr(df["avg_score"])

# Simple OLS
X_simple = sm.add_constant(df["str"])
model_simple = sm.OLS(df["avg_score"], X_simple).fit()

# OLS with common controls
controls = ["income", "lunch", "english", "expenditure"]
X_ctrl = sm.add_constant(df[["str"] + controls])
model_ctrl = sm.OLS(df["avg_score"], X_ctrl).fit()

# Save summary stats to stdout for inspection
print("Correlation (STR vs avg_score):", corr)
print("\nSimple OLS:")
print(model_simple.summary())
print("\nControlled OLS:")
print(model_ctrl.summary())
