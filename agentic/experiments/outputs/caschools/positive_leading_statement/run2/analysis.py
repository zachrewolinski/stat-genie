import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"

df = pd.read_csv(csv_path)

# Compute student-teacher ratio
# Avoid division by zero just in case

df = df.copy()
df["stratio"] = df["students"] / df["teachers"]

# Academic performance: average of read and math scores

df["perf"] = (df["read"] + df["math"]) / 2.0

# Drop missing values for relevant columns
basic_cols = ["perf", "stratio"]
reg_df = df[basic_cols].dropna()

# Simple correlation
corr = reg_df["perf"].corr(reg_df["stratio"])

# Simple OLS: perf ~ stratio
X = sm.add_constant(reg_df["stratio"])
model_simple = sm.OLS(reg_df["perf"], X).fit()

# Multiple OLS with common controls
control_cols = ["stratio", "lunch", "english", "income", "expenditure"]
reg_df2 = df[["perf"] + control_cols].dropna()
X2 = sm.add_constant(reg_df2[control_cols])
model_controls = sm.OLS(reg_df2["perf"], X2).fit()

print("Simple correlation (perf vs stratio):", corr)
print("\nSimple OLS: perf ~ stratio")
print(model_simple.summary())

print("\nOLS with controls: perf ~ stratio + lunch + english + income + expenditure")
print(model_controls.summary())
