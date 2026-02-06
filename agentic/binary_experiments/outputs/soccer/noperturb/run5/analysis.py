import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Skin tone (mean of two raters)
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin tone and games > 0
analysis_df = df.dropna(subset=["skin_tone", "games", "redCards"]).copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Dark vs light split at median
median_skin = analysis_df["skin_tone"].median()
analysis_df["dark_skin"] = (analysis_df["skin_tone"] >= median_skin).astype(int)

# Aggregate red-card rates by group
rate_table = (
    analysis_df.groupby("dark_skin")[["redCards", "games"]]
    .sum()
    .assign(red_per_100_games=lambda x: 100 * x["redCards"] / x["games"])
)

# Poisson regression with offset for exposure (games)
analysis_df["log_games"] = np.log(analysis_df["games"])

# Unadjusted model (skin tone only)
model_unadj = smf.glm(
    formula="redCards ~ skin_tone",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df["log_games"],
).fit()

# Adjusted model with basic controls
# Use categorical for position and leagueCountry, and numeric covariates
model_adj = smf.glm(
    formula="redCards ~ skin_tone + C(position) + C(leagueCountry) + height + weight + goals + yellowCards + yellowReds",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df["log_games"],
).fit()

# Collect key results
results = {
    "median_skin": float(median_skin),
    "rate_table": rate_table.reset_index().to_dict(orient="records"),
    "unadj_coef": float(model_unadj.params["skin_tone"]),
    "unadj_p": float(model_unadj.pvalues["skin_tone"]),
    "adj_coef": float(model_adj.params["skin_tone"]),
    "adj_p": float(model_adj.pvalues["skin_tone"]),
}

# Save a brief summary for inspection
summary_lines = []
summary_lines.append(f"Median skin tone: {results['median_skin']:.3f}")
summary_lines.append("Rates per 100 games by dark_skin (0=light,1=dark):")
for row in results["rate_table"]:
    summary_lines.append(
        f"  dark_skin={row['dark_skin']}: redCards={row['redCards']}, games={row['games']}, red_per_100_games={row['red_per_100_games']:.3f}"
    )
summary_lines.append(
    f"Unadjusted Poisson coef for skin_tone: {results['unadj_coef']:.4f} (p={results['unadj_p']:.4g})"
)
summary_lines.append(
    f"Adjusted Poisson coef for skin_tone: {results['adj_coef']:.4f} (p={results['adj_p']:.4g})"
)

with open("analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
