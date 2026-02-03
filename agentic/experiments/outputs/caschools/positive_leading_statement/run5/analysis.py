import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "/home/chenwang/stat-genie/agentic/experiments/outputs/caschools/positive_leading_statement/run5/caschools.csv"
df = pd.read_csv(path)

# Compute student-teacher ratio
# Avoid division by zero just in case

df = df.copy()
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of reading and math scores

df["avg_score"] = df[["read", "math"]].mean(axis=1)

# Basic correlation
corr = df[["str", "avg_score"]].corr().iloc[0, 1]

# Simple regression: avg_score ~ str
X1 = sm.add_constant(df[["str"]])
model1 = sm.OLS(df["avg_score"], X1, missing="drop").fit()

# Multiple regression controlling for demographics and spending
controls = ["lunch", "english", "income", "expenditure", "calworks", "computer"]
X2 = sm.add_constant(df[["str"] + controls])
model2 = sm.OLS(df["avg_score"], X2, missing="drop").fit()

# Save key results to a small report for inspection
report_lines = []
report_lines.append(f"Rows used: {len(df)}")
report_lines.append(f"Correlation(str, avg_score): {corr:.4f}")
report_lines.append("\nSimple OLS: avg_score ~ str")
report_lines.append(model1.summary().as_text())
report_lines.append("\nControlled OLS: avg_score ~ str + controls")
report_lines.append(model2.summary().as_text())

report_path = "/home/chenwang/stat-genie/agentic/experiments/outputs/caschools/positive_leading_statement/run5/analysis_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print("Analysis complete. Report written to", report_path)
