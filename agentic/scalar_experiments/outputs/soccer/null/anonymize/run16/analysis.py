import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Core variables
skin = df[["feature18", "feature19"]].mean(axis=1)

games = df["feature9"]
red = df["feature16"]

# Filter usable rows
mask = skin.notna() & games.notna() & red.notna() & (games > 0)
use = df.loc[mask].copy()
use["skin_mean"] = skin[mask]
use["games"] = games[mask]
use["red_cards"] = red[mask]

# Aggregate to player level (unique player)
player = (
    use.groupby("feature1", as_index=False)
    .agg(
        skin_mean=("skin_mean", "mean"),
        games=("games", "sum"),
        red_cards=("red_cards", "sum"),
    )
)

player = player[(player["games"] > 0) & player["skin_mean"].notna()].copy()

# Dark vs light definitions
player["dark_bin_075"] = np.where(player["skin_mean"] >= 0.75, "dark", np.where(player["skin_mean"] <= 0.25, "light", "mid"))
player["dark_bin_05"] = np.where(player["skin_mean"] > 0.5, "dark", np.where(player["skin_mean"] < 0.5, "light", "mid"))

# Aggregate rates for dark vs light
results = {}
for label in ["dark_bin_075", "dark_bin_05"]:
    sub = player[player[label].isin(["dark", "light"])].copy()
    agg = sub.groupby(label).agg(games=("games", "sum"), red_cards=("red_cards", "sum"))
    agg["rate_per_game"] = agg["red_cards"] / agg["games"]
    if set(agg.index) == {"dark", "light"}:
        rate_ratio = agg.loc["dark", "rate_per_game"] / agg.loc["light", "rate_per_game"]
    else:
        rate_ratio = np.nan
    results[label] = {
        "counts": sub[label].value_counts().to_dict(),
        "agg": agg.to_dict(),
        "rate_ratio": rate_ratio,
    }

# Poisson regression with exposure (player-level)
X = sm.add_constant(player["skin_mean"])
model = sm.GLM(player["red_cards"], X, family=sm.families.Poisson(), offset=np.log(player["games"]))
res = model.fit(cov_type="HC1")

beta = res.params["skin_mean"]
se = res.bse["skin_mean"]
pval = res.pvalues["skin_mean"]

# Interpretable rate ratio: dark (0.75) vs light (0.25)
rate_ratio_075_025 = float(np.exp(beta * (0.75 - 0.25)))

# Also compute player-level probability of any red card
player["any_red"] = (player["red_cards"] > 0).astype(int)
logit = sm.Logit(player["any_red"], X).fit(disp=False)
logit_beta = logit.params["skin_mean"]
logit_p = logit.pvalues["skin_mean"]

# Summaries
summary = {
    "n_rows_used": int(use.shape[0]),
    "n_players": int(player.shape[0]),
    "poisson_beta": float(beta),
    "poisson_se": float(se),
    "poisson_p": float(pval),
    "rate_ratio_075_025": rate_ratio_075_025,
    "logit_beta": float(logit_beta),
    "logit_p": float(logit_p),
    "group_rates": results,
}

print(json.dumps(summary, indent=2))
