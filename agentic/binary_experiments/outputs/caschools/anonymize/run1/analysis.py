import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"
df = pd.read_csv(csv_path)

# Map features based on metadata descriptions
# feature6: enrollment, feature7: teachers, feature14: reading, feature15: math

df = df.copy()
df["str"] = df["feature6"] / df["feature7"]  # student-teacher ratio

df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

# Simple correlation
corr = df[["str", "avg_score"]].corr().iloc[0, 1]

# OLS regression: avg_score ~ str
X = sm.add_constant(df["str"])
model = sm.OLS(df["avg_score"], X).fit()

# Save key results for conclusion
results = {
    "corr": corr,
    "coef_str": model.params["str"],
    "pvalue_str": model.pvalues["str"],
    "r2": model.rsquared,
}

# Print results for reference
print("Correlation (str vs avg_score):", results["corr"])
print("OLS coef (str):", results["coef_str"])
print("OLS p-value (str):", results["pvalue_str"])
print("OLS R^2:", results["r2"])

# Write a brief machine-readable summary
with open("analysis_results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}\t{v}\n")
