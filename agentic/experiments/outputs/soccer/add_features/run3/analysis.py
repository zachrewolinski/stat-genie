import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
DF_PATH = "soccer.csv"
df = pd.read_csv(DF_PATH)

# Compute mean skin tone using available raters
mean_tone = pd.concat([df["rater1"], df["rater2"]], axis=1).mean(axis=1, skipna=True)
df = df.assign(mean_tone=mean_tone)

# Define light vs dark groups based on the 5-point scale
# 0.0/0.25 = light, 0.5 = neutral, 0.75/1.0 = dark
mask = df["mean_tone"].notna() & ((df["mean_tone"] <= 0.25) | (df["mean_tone"] >= 0.75))
sub = df.loc[mask].copy()
sub = sub[sub["games"] > 0]
sub["dark"] = (sub["mean_tone"] >= 0.75).astype(int)

# Aggregate rates
rates = {}
for label, m in [("light", sub["dark"] == 0), ("dark", sub["dark"] == 1)]:
    red = sub.loc[m, "redCards"].sum()
    games = sub.loc[m, "games"].sum()
    rates[label] = {
        "red_cards": float(red),
        "games": float(games),
        "red_per_game": float(red / games) if games > 0 else float("nan"),
    }

# Poisson regression with offset log(games) for rate ratio
X = sm.add_constant(sub["dark"])
model = sm.GLM(sub["redCards"], X, family=sm.families.Poisson(), offset=np.log(sub["games"]))
res = model.fit()

rr = float(np.exp(res.params["dark"]))
ci = res.conf_int().loc["dark"]
rr_ci = (float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1])))
p_value = float(res.pvalues["dark"])

# Print a compact summary for inspection
print("Observations (light+dark dyads):", len(sub))
print("Rates:", rates)
print("Rate ratio (dark vs light):", rr)
print("95% CI:", rr_ci)
print("p-value:", p_value)
