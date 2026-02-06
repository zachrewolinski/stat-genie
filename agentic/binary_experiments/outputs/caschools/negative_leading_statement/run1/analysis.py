import pandas as pd
import statsmodels.api as sm

# Load data
path = "caschools.csv"
df = pd.read_csv(path)

# Student-teacher ratio and average score
# Higher ratio = more students per teacher (worse), lower ratio = fewer students per teacher (better)
df["str"] = df["students"] / df["teachers"]
df["avg_score"] = df[["read", "math"]].mean(axis=1)

# Simple correlation
corr = df["str"].corr(df["avg_score"])

# Simple OLS: average score on student-teacher ratio
X1 = sm.add_constant(df[["str"]])
model1 = sm.OLS(df["avg_score"], X1).fit()

# Controlled OLS with key demographics
controls = df[["str", "income", "lunch", "english"]].copy()
X2 = sm.add_constant(controls)
model2 = sm.OLS(df["avg_score"], X2).fit()

print("Correlation (student-teacher ratio vs avg score):", round(corr, 4))
print("\nModel 1: avg_score ~ str")
print(model1.summary().tables[1])
print("\nModel 2: avg_score ~ str + income + lunch + english")
print(model2.summary().tables[1])
