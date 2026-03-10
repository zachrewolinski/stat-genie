import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Skin tone: average of rater1 and rater2 (ignoring missing)
_df["skin"] = _df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Keep rows with skin and positive games
_df = _df.dropna(subset=["skin", "games", "redCards"]).copy()
_df = _df[_df["games"] > 0]

# Helper to fit Poisson with offset

def fit_poisson(df, formula):
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["games"]),
    )
    result = model.fit(cov_type="HC3")
    return result

# Dyad-level model
res_dyad = fit_poisson(_df, "redCards ~ skin")

# Player-level aggregation
player = (
    _df.groupby(["playerShort", "player"]).agg(
        games=("games", "sum"),
        redCards=("redCards", "sum"),
        skin=("skin", "mean"),
    )
    .reset_index()
)
player = player[player["games"] > 0]

res_player = fit_poisson(player, "redCards ~ skin")

# Group comparison: light vs dark (exclude mid-tone 0.5)
# light <= 0.25, dark >= 0.75
light = _df[_df["skin"] <= 0.25]
dark = _df[_df["skin"] >= 0.75]

# Compute rates per 100 games
light_rate = (light["redCards"].sum() / light["games"].sum()) * 100
dark_rate = (dark["redCards"].sum() / dark["games"].sum()) * 100

# Rate ratio with Poisson model on grouped data
# Build small grouped dataset
_group = pd.DataFrame(
    {
        "group": ["light", "dark"],
        "redCards": [light["redCards"].sum(), dark["redCards"].sum()],
        "games": [light["games"].sum(), dark["games"].sum()],
    }
)
_group["dark"] = (_group["group"] == "dark").astype(int)

res_group = smf.glm(
    "redCards ~ dark",
    data=_group,
    family=sm.families.Poisson(),
    offset=np.log(_group["games"]),
).fit()

# Collect key stats

def summarize_poisson(res):
    beta = res.params["skin"]
    se = res.bse["skin"]
    p = res.pvalues["skin"]
    irr = float(np.exp(beta))
    ci_low = float(np.exp(beta - 1.96 * se))
    ci_high = float(np.exp(beta + 1.96 * se))
    return {
        "beta": float(beta),
        "se": float(se),
        "p": float(p),
        "irr": irr,
        "irr_ci_low": ci_low,
        "irr_ci_high": ci_high,
    }

summary = {
    "n_rows": int(_df.shape[0]),
    "n_players": int(player.shape[0]),
    "dyad_model": summarize_poisson(res_dyad),
    "player_model": summarize_poisson(res_player),
    "light_rate_per_100": float(light_rate),
    "dark_rate_per_100": float(dark_rate),
    "group_rate_ratio": float(np.exp(res_group.params["dark"])),
    "group_rate_ratio_p": float(res_group.pvalues["dark"]),
    "light_games": float(light["games"].sum()),
    "dark_games": float(dark["games"].sum()),
    "light_redCards": float(light["redCards"].sum()),
    "dark_redCards": float(dark["redCards"].sum()),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
