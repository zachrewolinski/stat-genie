import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"
df = pd.read_csv(csv_path)

# Compute student-teacher ratio and average test score
# Avoid division by zero just in case

df = df.copy()
df["stratio"] = df["students"] / df["teachers"]
df["avg_score"] = (df["read"] + df["math"]) / 2.0

# Simple correlation
corr = df[["stratio", "avg_score"]].corr().iloc[0, 1]

# Simple OLS: avg_score ~ stratio
X_simple = sm.add_constant(df["stratio"])
model_simple = sm.OLS(df["avg_score"], X_simple).fit()

# Multiple OLS with common controls
controls = ["stratio", "income", "lunch", "english", "calworks", "expenditure"]
X_multi = sm.add_constant(df[controls])
model_multi = sm.OLS(df["avg_score"], X_multi).fit()

# Collect key results
results = {
    "corr_stratio_avgscore": corr,
    "simple_coef": model_simple.params["stratio"],
    "simple_pvalue": model_simple.pvalues["stratio"],
    "multi_coef": model_multi.params["stratio"],
    "multi_pvalue": model_multi.pvalues["stratio"],
    "n": len(df),
}

print("Correlation (stratio, avg_score):", results["corr_stratio_avgscore"])
print("Simple OLS coef (stratio):", results["simple_coef"], "p=", results["simple_pvalue"])
print("Multiple OLS coef (stratio):", results["multi_coef"], "p=", results["multi_pvalue"])
print("N:", results["n"])

# Save a short results summary for reference
with open("analysis_results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")
