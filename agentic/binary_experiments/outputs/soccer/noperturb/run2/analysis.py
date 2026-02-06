import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute mean skin tone from the two raters
skin_mean = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_mean=skin_mean)

# Keep rows with skin ratings and games > 0
analysis_df = df.dropna(subset=["skin_mean", "redCards", "games"]).copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Define skin tone groups (light vs dark)
# 0.0 = very light, 1.0 = very dark; use 0.5 split as light vs dark
analysis_df["dark"] = (analysis_df["skin_mean"] >= 0.5).astype(int)

# Descriptive stats: red card rates per game
summary = (
    analysis_df.groupby("dark").agg(
        n_dyads=("redCards", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
    )
)
summary["red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression for red card rate with games as exposure
# Model: redCards ~ dark, offset=log(games)
endog = analysis_df["redCards"]
exog = sm.add_constant(analysis_df["dark"])
offset = np.log(analysis_df["games"])

poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
poisson_results = poisson_model.fit()

# Extract effect for dark indicator
coef_dark = poisson_results.params["dark"]
se_dark = poisson_results.bse["dark"]
pval_dark = poisson_results.pvalues["dark"]
rate_ratio = np.exp(coef_dark)

print("Summary by dark (0=light, 1=dark):")
print(summary)
print("\nPoisson regression (red cards per game):")
print(f"coef_dark={coef_dark:.4f}, SE={se_dark:.4f}, p={pval_dark:.4g}, rate_ratio={rate_ratio:.4f}")
