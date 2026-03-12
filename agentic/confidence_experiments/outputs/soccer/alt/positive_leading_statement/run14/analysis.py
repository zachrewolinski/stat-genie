import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Skin tone: average of two raters
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)

# Keep rows with skin ratings and games > 0
base = df.dropna(subset=["skin_tone", "games", "redCards"]).copy()
base = base[base["games"] > 0]

# Binary dark vs light using midpoint 0.5; exclude neutrals
base = base[(base["skin_tone"] != 0.5)]
base = base.assign(dark=(base["skin_tone"] > 0.5).astype(int))

# Aggregate rates
agg = base.groupby("dark").agg(
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    n_rows=("redCards", "size"),
)
agg["red_per_10_games"] = 10 * agg["total_red"] / agg["total_games"]

rate_light = float(agg.loc[0, "red_per_10_games"]) if 0 in agg.index else np.nan
rate_dark = float(agg.loc[1, "red_per_10_games"]) if 1 in agg.index else np.nan
rate_ratio = (rate_dark / rate_light) if rate_light and not np.isnan(rate_dark) else np.nan

# Prepare datasets for models
base = base.assign(log_games=np.log(base["games"]))

# Unadjusted model dataset
unadj_cols = ["redCards", "dark", "log_games", "playerShort"]
base_unadj = base.dropna(subset=unadj_cols)

model_unadj = smf.glm(
    formula="redCards ~ dark",
    data=base_unadj,
    family=sm.families.Poisson(),
    offset=base_unadj["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": base_unadj["playerShort"]})

# Adjusted model dataset
for col in ["position", "leagueCountry"]:
    base[col] = base[col].astype("category")

adj_cols = [
    "redCards",
    "dark",
    "yellowCards",
    "yellowReds",
    "goals",
    "height",
    "weight",
    "position",
    "leagueCountry",
    "log_games",
    "playerShort",
]
base_adj = base.dropna(subset=adj_cols)

model_adj = smf.glm(
    formula="redCards ~ dark + yellowCards + yellowReds + goals + height + weight + C(position) + C(leagueCountry)",
    data=base_adj,
    family=sm.families.Poisson(),
    offset=base_adj["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": base_adj["playerShort"]})

# Negative binomial as robustness check (uses GLM with NB2)
model_nb = smf.glm(
    formula="redCards ~ dark + yellowCards + yellowReds + goals + height + weight + C(position) + C(leagueCountry)",
    data=base_adj,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=base_adj["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": base_adj["playerShort"]})

results = {
    "n_rows": int(base.shape[0]),
    "n_players": int(base["playerShort"].nunique()),
    "rate_light": rate_light,
    "rate_dark": rate_dark,
    "rate_ratio": rate_ratio,
    "unadj_coef": float(model_unadj.params["dark"]),
    "unadj_p": float(model_unadj.pvalues["dark"]),
    "unadj_rr": float(np.exp(model_unadj.params["dark"])),
    "adj_coef": float(model_adj.params["dark"]),
    "adj_p": float(model_adj.pvalues["dark"]),
    "adj_rr": float(np.exp(model_adj.params["dark"])),
    "nb_coef": float(model_nb.params["dark"]),
    "nb_p": float(model_nb.pvalues["dark"]),
    "nb_rr": float(np.exp(model_nb.params["dark"])),
    "n_rows_unadj": int(base_unadj.shape[0]),
    "n_rows_adj": int(base_adj.shape[0]),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
