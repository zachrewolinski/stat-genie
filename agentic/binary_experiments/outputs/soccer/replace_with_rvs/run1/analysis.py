import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("soccer.csv")

# Compute mean skin tone from two raters
_df["skin_mean"] = _df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin ratings and games
_df = _df.dropna(subset=["skin_mean", "games", "redCards"])

# Define light vs dark using thresholds around the 5-point scale
# Light: <= 1/3 (very light/light), Dark: >= 2/3 (dark/very dark)
_df["skin_group"] = np.where(_df["skin_mean"] >= (2.0/3.0), "dark",
                             np.where(_df["skin_mean"] <= (1.0/3.0), "light", "mid"))

# Focus on light vs dark
_df_ld = _df[_df["skin_group"].isin(["light", "dark"])].copy()

# Aggregate rates
agg = _df_ld.groupby("skin_group").agg(
    red_cards=("redCards", "sum"),
    games=("games", "sum"),
    dyads=("redCards", "size")
).reset_index()
agg["red_cards_per_game"] = agg["red_cards"] / agg["games"]

# Poisson regression with offset for games
_df_ld["dark"] = (_df_ld["skin_group"] == "dark").astype(int)

# Add small constant to avoid log(0) for games
_df_ld = _df_ld[_df_ld["games"] > 0]

X = sm.add_constant(_df_ld["dark"])
y = _df_ld["redCards"]
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(_df_ld["games"]))
result = model.fit()

# Extract incidence rate ratio for dark vs light
coef = result.params["dark"]
se = result.bse["dark"]
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# Save outputs for conclusion
agg.to_csv("analysis_agg.csv", index=False)

with open("analysis_summary.txt", "w") as f:
    f.write("Aggregate rates (red cards per game):\n")
    f.write(agg.to_string(index=False))
    f.write("\n\nPoisson regression (offset by games):\n")
    f.write(f"IRR (dark vs light) = {irr:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]\n")
    f.write(result.summary().as_text())
