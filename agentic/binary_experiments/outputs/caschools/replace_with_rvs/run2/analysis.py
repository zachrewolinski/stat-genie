import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"
df = pd.read_csv(csv_path)

# Compute student-teacher ratio
# Avoid division by zero just in case

df["stratio"] = df["students"] / df["teachers"]

# Academic performance: average of reading and math

df["avg_score"] = df[["read", "math"]].mean(axis=1)

# Drop rows with missing values in key columns
key_cols = ["stratio", "avg_score", "lunch", "income", "english", "expenditure"]
clean = df.dropna(subset=key_cols)

# Simple correlation
corr = clean[["stratio", "avg_score"]].corr().iloc[0, 1]

# Simple regression: avg_score ~ stratio
X_simple = sm.add_constant(clean["stratio"])
model_simple = sm.OLS(clean["avg_score"], X_simple).fit()

# Controlled regression: add socioeconomic/demographic controls
X_controls = clean[["stratio", "lunch", "income", "english", "expenditure"]]
X_controls = sm.add_constant(X_controls)
model_controls = sm.OLS(clean["avg_score"], X_controls).fit()

# Save key results for manual inspection
print("Rows used:", len(clean))
print("Correlation (stratio vs avg_score):", corr)
print("Simple regression coef (stratio):", model_simple.params["stratio"], "p=", model_simple.pvalues["stratio"])
print("Controlled regression coef (stratio):", model_controls.params["stratio"], "p=", model_controls.pvalues["stratio"])
