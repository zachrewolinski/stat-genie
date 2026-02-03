import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("soccer.csv")

# Compute mean skin tone (0-1 scale)
_df["skin_tone"] = _df[["rater1", "rater2"]].mean(axis=1)

# Define dark/light using midpoint 0.5; drop neutral/missing
_df = _df[_df["skin_tone"].notna()].copy()
_df = _df[_df["games"].notna() & (_df["games"] > 0)].copy()

_df["skin_group"] = np.where(_df["skin_tone"] > 0.5, "dark",
                      np.where(_df["skin_tone"] < 0.5, "light", "neutral"))
_df = _df[_df["skin_group"] != "neutral"].copy()

# Basic rate comparison
summary = (_df
           .groupby("skin_group")
           .agg(dyads=("redCards", "size"),
                total_games=("games", "sum"),
                total_reds=("redCards", "sum"))
          )
summary["reds_per_game"] = summary["total_reds"] / summary["total_games"]
summary["reds_per_10_games"] = summary["reds_per_game"] * 10

# Poisson regression with log(games) offset
# Model: redCards ~ dark indicator
_df["dark"] = (_df["skin_group"] == "dark").astype(int)

model = smf.glm(
    formula="redCards ~ dark",
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df["games"])
).fit(cov_type="HC0")

beta = model.params["dark"]
se = model.bse["dark"]
irr = float(np.exp(beta))
ci_low = float(np.exp(beta - 1.96 * se))
ci_high = float(np.exp(beta + 1.96 * se))

# Save key results for human-readable output
with open("analysis_results.txt", "w") as f:
    f.write("Summary by skin group (dyad-level):\n")
    f.write(summary.to_string())
    f.write("\n\n")
    f.write("Poisson regression (redCards ~ dark, offset log(games)):\n")
    f.write(model.summary().as_text())
    f.write("\n\n")
    f.write(f"IRR (dark vs light): {irr:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f})\n")
