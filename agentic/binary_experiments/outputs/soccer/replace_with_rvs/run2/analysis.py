import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("soccer.csv")

# Average skin tone across two raters
_df["skin_avg"] = (_df["rater1"] + _df["rater2"]) / 2

# Binary classification: light (<0.5) vs dark (>=0.5)
threshold = 0.5
_df["dark"] = (_df["skin_avg"] >= threshold).astype(int)

# Aggregate red-card rate per game by skin tone group
rate_dark = _df.loc[_df["dark"] == 1, "redCards"].sum() / _df.loc[_df["dark"] == 1, "games"].sum()
rate_light = _df.loc[_df["dark"] == 0, "redCards"].sum() / _df.loc[_df["dark"] == 0, "games"].sum()

# Poisson regression for red cards with exposure (games)
poisson_model = smf.glm(
    "redCards ~ dark",
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df["games"]),
)
poisson_res = poisson_model.fit()

rate_ratio = float(np.exp(poisson_res.params["dark"]))
rate_ratio_p = float(poisson_res.pvalues["dark"])

# Logistic regression for any red card (robustness check)
_df["any_red"] = (_df["redCards"] > 0).astype(int)
logit_model = smf.glm("any_red ~ dark + games", data=_df, family=sm.families.Binomial())
logit_res = logit_model.fit()
odds_ratio = float(np.exp(logit_res.params["dark"]))
odds_ratio_p = float(logit_res.pvalues["dark"])

print("Red-card rate per game (dark):", rate_dark)
print("Red-card rate per game (light):", rate_light)
print("Poisson rate ratio (dark vs light):", rate_ratio, "p=", rate_ratio_p)
print("Logit odds ratio (any red, dark vs light):", odds_ratio, "p=", odds_ratio_p)
