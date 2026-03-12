import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [c for c in df.columns if c.startswith("feature")]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="ignore")

# Define key variables
# feature4: masculinity-femininity index (1=masculine, 11=feminine)
# feature6: binary gender indicator (0 male, 1 female)
# feature8: deaths
# feature5: min pressure (lower = stronger)
# feature7: category
# feature13: max wind speed
# feature2: year

# Outcome
# Use log1p deaths to handle skew / zeros

df["log_deaths"] = np.log1p(df["feature8"])

# Controls
# Use log1p of max wind speed? It's already in mph; keep linear.
# Min pressure: lower is stronger; keep linear.
# Category: ordinal 1-5; keep linear.
# Year: to account for era differences in preparedness/forecasting.

controls = ["feature7", "feature5", "feature13", "feature2"]

# Build regression helper

def run_ols(y, x_cols, data):
    X = data[x_cols].copy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(data[y], X, missing="drop")
    res = model.fit()
    return res

results = {}

# Model 1: bivariate log deaths ~ femininity index
res1 = run_ols("log_deaths", ["feature4"], df)
results["model1"] = res1

# Model 2: controls
res2 = run_ols("log_deaths", ["feature4"] + controls, df)
results["model2"] = res2

# Model 3: binary gender instead of index
res3 = run_ols("log_deaths", ["feature6"] + controls, df)
results["model3"] = res3

# Model 4: add log1p property damage (feature14) as additional control (robustness)
# Some missing values in feature14; drop accordingly.
df["log_damage"] = np.log1p(df["feature14"])
res4 = run_ols("log_deaths", ["feature4"] + controls + ["log_damage"], df)
results["model4"] = res4

# Summary stats
summary = {
    "n_total": int(df.shape[0]),
    "deaths_mean": float(df["feature8"].mean()),
    "deaths_median": float(df["feature8"].median()),
    "deaths_zero_count": int((df["feature8"] == 0).sum()),
    "fem_index_mean": float(df["feature4"].mean()),
    "fem_index_std": float(df["feature4"].std()),
    "female_binary_mean": float(df["feature6"].mean()),
}

print("SUMMARY", summary)
for name, res in results.items():
    print("\n", name)
    print(res.summary())

# Also compute simple correlations
corr = df[["feature4", "feature6", "feature8", "log_deaths", "feature7", "feature5", "feature13"]].corr(numeric_only=True)
print("\nCORR\n", corr)
