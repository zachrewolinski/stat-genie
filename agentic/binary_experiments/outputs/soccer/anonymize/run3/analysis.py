import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute average skin tone from two raters
skin = df[["feature18", "feature19"]].mean(axis=1)

# Define light vs dark; exclude middle category (0.5) for clearer contrast
light_mask = skin < 0.5
dark_mask = skin > 0.5

# Keep rows with clear light or dark
mask = light_mask | dark_mask
sub = df.loc[mask].copy()
sub["dark"] = (skin.loc[mask] > 0.5).astype(int)

# Outcomes
sub["games"] = sub["feature9"].astype(float)
sub["red_cards"] = sub["feature16"].astype(float)

# Basic rates
rate_light = (sub.loc[sub["dark"] == 0, "red_cards"].sum() /
              sub.loc[sub["dark"] == 0, "games"].sum())
rate_dark = (sub.loc[sub["dark"] == 1, "red_cards"].sum() /
             sub.loc[sub["dark"] == 1, "games"].sum())

# Poisson regression with exposure offset
X = sm.add_constant(sub["dark"])
model = sm.GLM(sub["red_cards"], X, family=sm.families.Poisson(),
               offset=np.log(sub["games"]))
res = model.fit()
coef = res.params["dark"]
pval = res.pvalues["dark"]
rate_ratio = np.exp(coef)

print("Rows used:", len(sub))
print("Total light dyads:", (sub["dark"] == 0).sum())
print("Total dark dyads:", (sub["dark"] == 1).sum())
print("Red cards per game (light):", rate_light)
print("Red cards per game (dark):", rate_dark)
print("Rate ratio (dark vs light):", rate_ratio)
print("P-value:", pval)
