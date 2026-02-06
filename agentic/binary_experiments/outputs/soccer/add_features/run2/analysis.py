import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DF_PATH = "soccer.csv"
df = pd.read_csv(DF_PATH)

# Skin tone: average of two raters (0=very light, 1=very dark)
df["skin_avg"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin ratings and games > 0
analysis_df = df.dropna(subset=["skin_avg", "redCards", "games"]).copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Define dark vs light: dark if average > 0.5, light if <= 0.5
analysis_df["dark"] = (analysis_df["skin_avg"] > 0.5).astype(int)

# Rate per game
analysis_df["red_rate"] = analysis_df["redCards"] / analysis_df["games"]

# Group comparison
summary = analysis_df.groupby("dark").agg(
    n=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    mean_rate=("red_rate", "mean"),
)
summary["rate_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression with exposure offset
# redCards ~ dark + offset(log(games))
X = sm.add_constant(analysis_df["dark"])
model = sm.GLM(
    analysis_df["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df["games"]),
)
res = model.fit(cov_type="HC0")

# Extract rate ratio for dark vs light
coef = res.params["dark"]
se = res.bse["dark"]
rate_ratio = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# Save key outputs for inspection
print("Group summary (0=light, 1=dark):")
print(summary)
print("\nPoisson regression with offset(log(games)):")
print(res.summary())
print("\nRate ratio (dark vs light):", rate_ratio)
print("95% CI:", (ci_low, ci_high))
