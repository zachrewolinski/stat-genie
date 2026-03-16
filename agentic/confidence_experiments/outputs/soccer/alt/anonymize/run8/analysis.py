import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "soccer.csv"
df = pd.read_csv(DATA_PATH)

# Columns
skin1 = df["feature18"]
skin2 = df["feature19"]
# Average skin tone; keep NaN if both missing
skin = pd.concat([skin1, skin2], axis=1).mean(axis=1)

# Outcome and exposure
red_cards = df["feature16"]
games = df["feature9"]

# Keep valid rows
valid = skin.notna() & red_cards.notna() & games.notna() & (games > 0)
work = df.loc[valid].copy()
work["skin"] = skin.loc[valid]
work["red_cards"] = red_cards.loc[valid]
work["games"] = games.loc[valid]
work["red_any"] = (work["red_cards"] > 0).astype(int)
work["red_rate"] = work["red_cards"] / work["games"]

# Define dark/light groups using 5-point scale normalized to [0,1]
# Map: 0, 0.25, 0.5, 0.75, 1.0
work["dark_group"] = np.where(work["skin"] >= 0.75, 1, np.where(work["skin"] <= 0.25, 0, np.nan))

dark_light = work.dropna(subset=["dark_group"]).copy()

# Helper: fit Poisson with robust SE

def fit_poisson(endog, exog, offset):
    model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type="HC1")
    return res

results = {}

# Model 1: continuous skin tone
exog1 = sm.add_constant(work["skin"])
res1 = fit_poisson(work["red_cards"], exog1, np.log(work["games"]))
coef_skin = res1.params["skin"]
se_skin = res1.bse["skin"]
p_skin = res1.pvalues["skin"]
irr_skin = float(np.exp(coef_skin))
ci_low, ci_high = np.exp(res1.conf_int().loc["skin"])

results["poisson_continuous"] = {
    "coef": float(coef_skin),
    "se": float(se_skin),
    "p": float(p_skin),
    "irr": float(irr_skin),
    "irr_ci_low": float(ci_low),
    "irr_ci_high": float(ci_high),
    "n": int(work.shape[0]),
}

# Model 2: dark vs light subset
exog2 = sm.add_constant(dark_light["dark_group"])
res2 = fit_poisson(dark_light["red_cards"], exog2, np.log(dark_light["games"]))
coef_dark = res2.params["dark_group"]
se_dark = res2.bse["dark_group"]
p_dark = res2.pvalues["dark_group"]
irr_dark = float(np.exp(coef_dark))
ci_low2, ci_high2 = np.exp(res2.conf_int().loc["dark_group"])

results["poisson_dark_vs_light"] = {
    "coef": float(coef_dark),
    "se": float(se_dark),
    "p": float(p_dark),
    "irr": float(irr_dark),
    "irr_ci_low": float(ci_low2),
    "irr_ci_high": float(ci_high2),
    "n": int(dark_light.shape[0]),
    "dark_n": int(dark_light["dark_group"].sum()),
    "light_n": int((dark_light["dark_group"] == 0).sum()),
}

# Descriptive rates
work["red_rate"] = work["red_cards"] / work["games"]
rate_dark = dark_light.loc[dark_light["dark_group"] == 1, "red_rate"].mean()
rate_light = dark_light.loc[dark_light["dark_group"] == 0, "red_rate"].mean()

results["rates"] = {
    "mean_rate_dark": float(rate_dark),
    "mean_rate_light": float(rate_light),
    "rate_ratio": float(rate_dark / rate_light) if rate_light > 0 else float("inf"),
}

# Logistic regression for red_any (sensitivity)
# include log(games) as covariate
exog3 = sm.add_constant(pd.DataFrame({
    "skin": work["skin"],
    "log_games": np.log(work["games"]),
}))
logit = sm.Logit(work["red_any"], exog3)
res3 = logit.fit(disp=False)
coef_logit = res3.params["skin"]
p_logit = res3.pvalues["skin"]
orr = float(np.exp(coef_logit))
ci_low3, ci_high3 = np.exp(res3.conf_int().loc["skin"])
results["logit_any"] = {
    "coef": float(coef_logit),
    "p": float(p_logit),
    "or": float(orr),
    "or_ci_low": float(ci_low3),
    "or_ci_high": float(ci_high3),
    "n": int(work.shape[0]),
}

# Save intermediate results for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
