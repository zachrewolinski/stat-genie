import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute skin tone as mean of rater1 and rater2
# If one missing, use the other; if both missing, NaN
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Keep rows with skin tone and games > 0
analysis_df = df[(df["games"] > 0) & df["skin_tone"].notna()].copy()

# Create per-game red card rate
analysis_df["red_rate"] = analysis_df["redCards"] / analysis_df["games"]

# Dichotomize skin tone: light vs dark using median split (0.5 threshold corresponds to mid scale)
analysis_df["dark_skin"] = (analysis_df["skin_tone"] > 0.5).astype(int)

# Summary stats
summary = {
    "n_rows": len(analysis_df),
    "n_players": analysis_df["playerShort"].nunique(),
    "mean_skin": analysis_df["skin_tone"].mean(),
    "mean_red_rate": analysis_df["red_rate"].mean(),
    "mean_red_rate_dark": analysis_df.loc[analysis_df["dark_skin"] == 1, "red_rate"].mean(),
    "mean_red_rate_light": analysis_df.loc[analysis_df["dark_skin"] == 0, "red_rate"].mean(),
}

# Poisson regression with offset for games
# Model 1: skin tone continuous
analysis_df["log_games"] = np.log(analysis_df["games"])
# Model 1 uses all rows with skin_tone and games
model1 = smf.glm(
    formula="redCards ~ skin_tone",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": analysis_df["playerShort"]})

# Model 2: with controls (position, leagueCountry)
analysis_df_m2 = analysis_df.dropna(subset=["position", "leagueCountry", "playerShort"])
model2 = smf.glm(
    formula="redCards ~ skin_tone + C(position) + C(leagueCountry)",
    data=analysis_df_m2,
    family=sm.families.Poisson(),
    offset=analysis_df_m2["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": analysis_df_m2["playerShort"]})

# Model 3: binary dark_skin
model3 = smf.glm(
    formula="redCards ~ dark_skin",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": analysis_df["playerShort"]})


def extract(model, var):
    coef = model.params[var]
    se = model.bse[var]
    p = model.pvalues[var]
    rr = np.exp(coef)
    return {"coef": coef, "se": se, "p": p, "rr": rr}

results = {
    "summary": summary,
    "model1_skin_tone": extract(model1, "skin_tone"),
    "model2_skin_tone": extract(model2, "skin_tone"),
    "model3_dark_skin": extract(model3, "dark_skin"),
}

# Save results
out = Path(__file__).resolve().parent / "analysis_results.json"
import json
with out.open("w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
