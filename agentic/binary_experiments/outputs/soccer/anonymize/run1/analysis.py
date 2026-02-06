import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone: average of two raters
skin_avg = df[["feature18", "feature19"]].mean(axis=1)

df = df.copy()
df["skin_avg"] = skin_avg

# Define light vs dark using clear cutoffs around the midpoint (0.5)
# Light: <= 0.4, Dark: >= 0.6; exclude middle to reduce ambiguity
light_mask = df["skin_avg"] <= 0.4
dark_mask = df["skin_avg"] >= 0.6

subset = df[light_mask | dark_mask].copy()
subset["dark"] = np.where(subset["skin_avg"] >= 0.6, 1, 0)

# Outcome and exposure
subset["red_cards"] = subset["feature16"].astype(float)
subset["games"] = subset["feature9"].astype(float)

# Avoid zero exposure (shouldn't happen, but guard)
subset = subset[subset["games"] > 0]

# Poisson regression with log(games) offset to model red card rate per game
X = sm.add_constant(subset["dark"])
model = sm.GLM(
    subset["red_cards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(subset["games"]),
)
result = model.fit()

# Rate ratio for dark vs light
rate_ratio = np.exp(result.params["dark"])
ci = result.conf_int().loc["dark"]
ci_rr = np.exp(ci)

# Compute empirical rates per game for transparency
rate_light = (subset.loc[subset["dark"] == 0, "red_cards"].sum() /
              subset.loc[subset["dark"] == 0, "games"].sum())
rate_dark = (subset.loc[subset["dark"] == 1, "red_cards"].sum() /
             subset.loc[subset["dark"] == 1, "games"].sum())

summary = {
    "n_total": len(df),
    "n_light_dark": len(subset),
    "rate_light": rate_light,
    "rate_dark": rate_dark,
    "rate_ratio": rate_ratio,
    "ci_low": ci_rr[0],
    "ci_high": ci_rr[1],
    "p_value": result.pvalues["dark"],
}

print(summary)
