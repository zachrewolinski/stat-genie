import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute average skin tone rating
skin_avg = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_avg=skin_avg)

# Focus on rows with skin ratings and valid games
analysis_df = df.dropna(subset=["skin_avg", "games", "redCards"]).copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Define dark vs light: dark if skin_avg >= 0.5 (darker than neutral midpoint)
analysis_df["dark"] = (analysis_df["skin_avg"] >= 0.5).astype(int)

# Compute red card rate per game for summary
analysis_df["red_rate"] = analysis_df["redCards"] / analysis_df["games"]

summary = analysis_df.groupby("dark").agg(
    dyads=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    mean_red_rate=("red_rate", "mean"),
).reset_index()
summary["rate_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression with exposure (games) to compare rates
analysis_df["log_games"] = np.log(analysis_df["games"])
model = smf.glm(
    formula="redCards ~ dark",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df["log_games"],
).fit(cov_type="HC0")

coef = model.params["dark"]
se = model.bse["dark"]
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

print("Dark vs Light summary (dark=1, light=0):")
print(summary)
print("\nPoisson regression (redCards ~ dark, offset=log(games))")
print(model.summary().tables[1])
print(f"\nIncidence Rate Ratio (dark vs light): {irr:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f})")
