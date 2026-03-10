import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone (0-1 scale)
df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

# Drop rows without skin rating or games
base = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()

# Ensure numeric
for col in ["games", "redCards", "yellowCards", "yellowReds", "goals", "victories", "ties", "defeats"]:
    base[col] = pd.to_numeric(base[col], errors="coerce")

base = base.dropna(subset=["games", "redCards"])
base = base[base["games"] > 0]

# Skin tone categories based on discrete scale (0, 0.25, 0.5, 0.75, 1)
# Light: 0 or 0.25; Medium: 0.5; Dark: 0.75 or 1
bins = [-0.01, 0.26, 0.51, 1.01]
labels = ["light", "medium", "dark"]
base["skin_cat"] = pd.cut(base["skin_mean"], bins=bins, labels=labels)

# Summary stats by category
summary = base.groupby("skin_cat").agg(
    dyads=("redCards", "size"),
    players=("playerShort", "nunique"),
    total_games=("games", "sum"),
    total_red=("redCards", "sum"),
    red_rate_per_game=("redCards", lambda x: x.sum() / base.loc[x.index, "games"].sum()),
    red_any_rate=("redCards", lambda x: (x > 0).mean())
)

# Poisson regression with offset log(games)
# Use skin_mean as continuous predictor
# Include controls for leagueCountry, position, and goals/yellows as proxies for aggressiveness
# Keep it modest to avoid overfitting
base["log_games"] = np.log(base["games"])

# Model 1: skin_mean only
model1 = smf.glm(
    formula="redCards ~ skin_mean",
    data=base,
    family=sm.families.Poisson(),
    offset=base["log_games"],
).fit(cov_type="HC3")

# Model 2: with league and position fixed effects + performance/discipline controls
# Some categorical values may be missing; drop rows with missing categories
model2_data = base.dropna(subset=["leagueCountry", "position", "yellowCards", "yellowReds", "goals"])
model2 = smf.glm(
    formula="redCards ~ skin_mean + C(leagueCountry) + C(position) + yellowCards + yellowReds + goals",
    data=model2_data,
    family=sm.families.Poisson(),
    offset=model2_data["log_games"],
).fit(cov_type="HC3")

# Model 3: skin category (light baseline)
model3_data = base.dropna(subset=["skin_cat"])
model3 = smf.glm(
    formula="redCards ~ C(skin_cat)",
    data=model3_data,
    family=sm.families.Poisson(),
    offset=model3_data["log_games"],
).fit(cov_type="HC3")

# Logistic model for any red card
base["any_red"] = (base["redCards"] > 0).astype(int)
logit1 = smf.glm(
    formula="any_red ~ skin_mean",
    data=base,
    family=sm.families.Binomial(),
).fit(cov_type="HC3")

# Prepare outputs
print("Rows with skin rating:", len(base))
print("Skin mean unique values:", sorted(base["skin_mean"].unique())[:10], "... total", base["skin_mean"].nunique())
print("\nSummary by skin category:")
print(summary)

print("\nPoisson model 1 (skin_mean only):")
print(model1.summary().tables[1])

print("\nPoisson model 2 (with controls):")
print(model2.summary().tables[1])

print("\nPoisson model 3 (skin category):")
print(model3.summary().tables[1])

print("\nLogit model (any red):")
print(logit1.summary().tables[1])

# Effect size: rate ratio for skin_mean from 0 to 1
coef1 = model1.params["skin_mean"]
rr1 = np.exp(coef1)
coef2 = model2.params["skin_mean"]
rr2 = np.exp(coef2)

print("\nRate ratio (skin_mean 0->1):")
print("Model1 RR:", rr1)
print("Model2 RR:", rr2)

# Simple difference in red rate per game light vs dark
light = summary.loc["light"]
dark = summary.loc["dark"]
print("\nRed rate per game light:", light["red_rate_per_game"], "dark:", dark["red_rate_per_game"])
print("Red any rate light:", light["red_any_rate"], "dark:", dark["red_any_rate"])
