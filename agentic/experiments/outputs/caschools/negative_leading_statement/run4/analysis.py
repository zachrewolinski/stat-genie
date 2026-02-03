import pandas as pd
import statsmodels.api as sm

# Load data
path = "caschools.csv"
df = pd.read_csv(path)

# Construct student-teacher ratio and academic performance
# Student-teacher ratio: students per teacher
# Academic performance: average of reading and math scores

df = df.copy()
df["stratio"] = df["students"] / df["teachers"]
df["avg_score"] = (df["read"] + df["math"]) / 2

# Basic correlation
corr = df["stratio"].corr(df["avg_score"])

# Simple linear regression: avg_score ~ stratio
X1 = sm.add_constant(df[["stratio"]])
model1 = sm.OLS(df["avg_score"], X1).fit()

# Multiple regression with common controls
controls = ["income", "english", "lunch", "expenditure"]
X2 = sm.add_constant(df[["stratio"] + controls])
model2 = sm.OLS(df["avg_score"], X2).fit()

# Print key results for inspection
print("Correlation (stratio vs avg_score):", corr)
print("\nSimple regression: avg_score ~ stratio")
print(model1.summary().tables[1])
print("\nMultiple regression: avg_score ~ stratio + controls")
print(model2.summary().tables[1])
