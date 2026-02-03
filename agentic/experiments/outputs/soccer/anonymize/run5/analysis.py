import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average (two raters). Use mean when both available.
df["skin_avg"] = df[["feature18", "feature19"]].mean(axis=1)

# Define dark vs light using 5-point scale normalized to 0-1
# Values are {0, 0.25, 0.5, 0.75, 1}. Treat 0.5 as neutral and exclude.
df["skin_group"] = np.where(df["skin_avg"] > 0.5, "dark",
                      np.where(df["skin_avg"] < 0.5, "light", "neutral"))

# Outcomes and exposure
red = df["feature16"].astype(float)
games = df["feature9"].astype(float)

# Filter to light/dark only and positive games
mask = df["skin_group"].isin(["light", "dark"]) & (games > 0) & red.notna() & df["skin_avg"].notna()
df2 = df.loc[mask].copy()

# Descriptive rates
rates = df2.groupby("skin_group").apply(lambda g: g["feature16"].sum() / g["feature9"].sum())

# Poisson regression with log(games) offset
# Response: red cards count; Predictor: dark vs light
X = (df2["skin_group"] == "dark").astype(int)
X = sm.add_constant(X)
model = sm.GLM(df2["feature16"], X, family=sm.families.Poisson(), offset=np.log(df2["feature9"]))
result = model.fit()

# Extract rate ratio for dark vs light
coef = result.params[1]
rr = float(np.exp(coef))

# Two-sided p-value for dark coefficient
p_value = float(result.pvalues[1])

# Save a small summary for reference
summary_lines = [
    f"dark_rate={rates.get('dark', np.nan):.6f}",
    f"light_rate={rates.get('light', np.nan):.6f}",
    f"rate_ratio_dark_vs_light={rr:.4f}",
    f"p_value={p_value:.6g}",
    f"n_rows={len(df2)}",
]

with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))
