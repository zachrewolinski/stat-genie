import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv("soccer.csv")

# Compute mean skin tone rating (0-1 scale, 5 discrete values)
_df["skin_mean"] = _df[["rater1", "rater2"]].mean(axis=1)

# Keep only rows with skin ratings and games > 0
_df = _df[_df["skin_mean"].notna()].copy()
_df = _df[_df["games"] > 0].copy()

# Define light vs dark (exclude the neutral midpoint 0.5 for a cleaner contrast)
_df["skin_cat"] = np.where(_df["skin_mean"] > 0.5, "dark", np.where(_df["skin_mean"] < 0.5, "light", "mid"))
_df = _df[_df["skin_cat"].isin(["light", "dark"])].copy()

# Aggregate to player level to avoid repeated dyads dominating the signal
player = (
    _df.groupby("playerShort", as_index=False)
    .agg(
        skin_cat=("skin_cat", "first"),
        skin_mean=("skin_mean", "mean"),
        games=("games", "sum"),
        redCards=("redCards", "sum"),
    )
)

# Summaries
summary = {}
for cat in ["light", "dark"]:
    sub = player[player["skin_cat"] == cat]
    summary[cat] = {
        "players": int(sub.shape[0]),
        "total_games": float(sub["games"].sum()),
        "total_red": float(sub["redCards"].sum()),
        "rate_per_game": float(sub["redCards"].sum() / sub["games"].sum()),
    }

# Poisson regression at player level with exposure offset
X = (player["skin_cat"] == "dark").astype(int)
X = sm.add_constant(X)

model = sm.GLM(
    player["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(player["games"]),
)
result = model.fit()
# Robust SEs for potential overdispersion/heteroskedasticity
robust = model.fit(cov_type="HC3")

coef = robust.params[1]
se = robust.bse[1]
pval = robust.pvalues[1]
rate_ratio = float(np.exp(coef))

# Dyad-level check with clustered SE by player
_df["dark"] = (_df["skin_cat"] == "dark").astype(int)
X2 = sm.add_constant(_df["dark"])
model2 = sm.GLM(
    _df["redCards"],
    X2,
    family=sm.families.Poisson(),
    offset=np.log(_df["games"]),
)
result2 = model2.fit(cov_type="cluster", cov_kwds={"groups": _df["playerShort"]})
coef2 = result2.params[1]
pval2 = result2.pvalues[1]
rate_ratio2 = float(np.exp(coef2))

out = {
    "summary": summary,
    "player_level": {
        "coef_log_rr": float(coef),
        "se": float(se),
        "pval": float(pval),
        "rate_ratio": rate_ratio,
    },
    "dyad_level_cluster": {
        "coef_log_rr": float(coef2),
        "pval": float(pval2),
        "rate_ratio": rate_ratio2,
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)
