import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone: average of the two raters
skin_cols = ["rater1", "rater2"]
df["skin"] = df[skin_cols].mean(axis=1)

# Basic cleaning
base = df.dropna(subset=["skin", "redCards", "games"]).copy()
base = base[base["games"] > 0]

# Rate per game for descriptive stats
base["red_rate"] = base["redCards"] / base["games"]

# Skin tone groups: light vs dark (using extremes)
# 5-point scale normalized to [0,1] with 0,0.25,0.5,0.75,1.0
# light: <=0.25; dark: >=0.75
base["skin_group"] = np.where(
    base["skin"] <= 0.25,
    "light",
    np.where(base["skin"] >= 0.75, "dark", "mid"),
)

group_stats = (
    base.groupby("skin_group")
    .agg(
        n=("skin", "size"),
        red_cards=("redCards", "sum"),
        games=("games", "sum"),
        red_rate=("red_rate", "mean"),
    )
    .reset_index()
)

# Poisson regression with offset for games
model1 = smf.glm(
    formula="redCards ~ skin",
    data=base,
    family=sm.families.Poisson(),
    offset=np.log(base["games"]),
).fit(cov_type="HC3")

# Model with controls (position, leagueCountry, height, weight)
control_cols = ["position", "leagueCountry", "height", "weight"]
base_controls = base.dropna(subset=control_cols).copy()
model2 = smf.glm(
    formula="redCards ~ skin + C(position) + C(leagueCountry) + height + weight",
    data=base_controls,
    family=sm.families.Poisson(),
    offset=np.log(base_controls["games"]),
).fit(cov_type="HC3")

# Dark vs light only comparison
extreme = base[base["skin_group"].isin(["light", "dark"])].copy()
extreme["dark"] = (extreme["skin_group"] == "dark").astype(int)
model3 = smf.glm(
    formula="redCards ~ dark",
    data=extreme,
    family=sm.families.Poisson(),
    offset=np.log(extreme["games"]),
).fit(cov_type="HC3")


def extract_effect(model, term):
    coef = model.params[term]
    se = model.bse[term]
    pval = model.pvalues[term]
    ci_low, ci_high = model.conf_int().loc[term]
    irr = float(np.exp(coef))
    irr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "irr": irr,
        "irr_ci": irr_ci,
    }

results = {
    "n_total": int(base.shape[0]),
    "group_stats": group_stats.to_dict(orient="records"),
    "model1_skin": extract_effect(model1, "skin"),
    "model2_skin": extract_effect(model2, "skin"),
    "model3_dark": extract_effect(model3, "dark"),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
