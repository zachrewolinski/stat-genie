import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute mean skin tone (0=very light, 1=very dark)
# Use available ratings; drop rows without ratings
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

df = df.copy()
# if both are NaN, skin will be NaN
skin = skin.where(~skin.isna())

df["skin_mean"] = skin

# Keep rows with skin tone and games>0
analysis_df = df[(~df["skin_mean"].isna()) & (df["games"] > 0)].copy()

# Define light vs dark: light <= 0.5, dark > 0.5
analysis_df["dark_skin"] = (analysis_df["skin_mean"] > 0.5).astype(int)

# Compute red cards per game
analysis_df["red_per_game"] = analysis_df["redCards"] / analysis_df["games"]

# Group summary
summary = analysis_df.groupby("dark_skin").agg(
    n_dyads=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    mean_red_per_game=("red_per_game", "mean"),
)
summary["rate_red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression with offset for games to model red card counts per game
# redCards ~ dark_skin
X = sm.add_constant(analysis_df["dark_skin"])
model = sm.GLM(
    analysis_df["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df["games"]),
)
result = model.fit()

coef = result.params["dark_skin"]
pval = result.pvalues["dark_skin"]
rate_ratio = float(np.exp(coef))

# Save key results to stdout for debugging if run directly
if __name__ == "__main__":
    print("Group summary:\n", summary)
    print("Poisson dark_skin coef:", coef)
    print("Rate ratio (dark vs light):", rate_ratio)
    print("p-value:", pval)
