import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("soccer.csv")

# Skin tone average from two raters
skin = df[["rater1", "rater2"]].mean(axis=1)

df = df.assign(skin=skin)

# Basic cleaning
_df = df[(df["games"] > 0) & df["skin"].notna()].copy()

# Group comparison: light (<=0.25) vs dark (>=0.75)
light_mask = _df["skin"] <= 0.25
dark_mask = _df["skin"] >= 0.75

_df_group = _df[light_mask | dark_mask].copy()
_df_group["dark"] = (_df_group["skin"] >= 0.75).astype(int)

# Aggregate rates
agg = _df_group.groupby("dark").agg(
    red_cards=("redCards", "sum"),
    games=("games", "sum"),
    dyads=("redCards", "size"),
)
agg["rate_per_game"] = agg["red_cards"] / agg["games"]

# Poisson rate ratio (dark vs light) using GLM with offset
X = sm.add_constant(_df_group["dark"])
offset = np.log(_df_group["games"])
poisson_model = sm.GLM(_df_group["redCards"], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC0")

coef_dark = poisson_res.params["dark"]
se_dark = poisson_res.bse["dark"]
rr_dark = math.exp(coef_dark)
ci_low = math.exp(coef_dark - 1.96 * se_dark)
ci_high = math.exp(coef_dark + 1.96 * se_dark)
pval_dark = poisson_res.pvalues["dark"]

# Continuous skin tone model
Xc = sm.add_constant(_df["skin"])
offset_c = np.log(_df["games"])
poisson_c = sm.GLM(_df["redCards"], Xc, family=sm.families.Poisson(), offset=offset_c)
poisson_c_res = poisson_c.fit(cov_type="HC0")

coef_skin = poisson_c_res.params["skin"]
se_skin = poisson_c_res.bse["skin"]
rr_skin = math.exp(coef_skin)
ci_skin_low = math.exp(coef_skin - 1.96 * se_skin)
ci_skin_high = math.exp(coef_skin + 1.96 * se_skin)
pval_skin = poisson_c_res.pvalues["skin"]

# Effect for a 0.25 increase (one scale step)
rr_skin_step = math.exp(coef_skin * 0.25)

# Prepare results for explanation
light_stats = agg.loc[0].to_dict() if 0 in agg.index else {}
dark_stats = agg.loc[1].to_dict() if 1 in agg.index else {}

results = {
    "light": light_stats,
    "dark": dark_stats,
    "rr_dark_vs_light": rr_dark,
    "rr_dark_ci": [ci_low, ci_high],
    "pval_dark": pval_dark,
    "rr_skin_fullscale": rr_skin,
    "rr_skin_fullscale_ci": [ci_skin_low, ci_skin_high],
    "pval_skin": pval_skin,
    "rr_skin_per_0_25": rr_skin_step,
}

with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Decide response scale based on significance and direction
# Start at neutral 50, shift based on evidence.
response = 50

if pval_skin < 0.05 and coef_skin > 0:
    # evidence for higher red cards with darker skin
    # scale by effect size: small ~60-70, moderate ~70-80
    if rr_skin >= 1.25:
        response = 80
    elif rr_skin >= 1.10:
        response = 70
    else:
        response = 60
elif pval_skin < 0.05 and coef_skin < 0:
    # evidence for lower red cards with darker skin
    if rr_skin <= 0.80:
        response = 20
    elif rr_skin <= 0.90:
        response = 30
    else:
        response = 40
else:
    # no significant evidence
    response = 40 if coef_skin >= 0 else 60

# Build explanation text
light_rate = light_stats.get("rate_per_game", float("nan"))
dark_rate = dark_stats.get("rate_per_game", float("nan"))

explanation = (
    "I tested whether darker skin tone is associated with a higher red-card rate using Poisson models "
    "with an exposure offset for games played. The continuous skin-tone model (0–1 scale) yielded a "
    f"rate ratio of {rr_skin:.3f} for a full-scale increase (95% CI {ci_skin_low:.3f}–{ci_skin_high:.3f}, "
    f"p={pval_skin:.3g}), which corresponds to a per-step (0.25) rate ratio of {rr_skin_step:.3f}. "
    "A grouped comparison of very light (<=0.25) vs very dark (>=0.75) players produced a dark-vs-light "
    f"rate ratio of {rr_dark:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f}, p={pval_dark:.3g}). "
    f"Observed red-card rates per game were {light_rate:.5f} for light players and {dark_rate:.5f} for dark players. "
    "Taken together, these results indicate the strength and direction of the relationship, and the Likert response "
    "reflects the statistical significance and effect size." 
)

conclusion = {
    "response": int(response),
    "explanation": explanation,
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(conclusion, f)
