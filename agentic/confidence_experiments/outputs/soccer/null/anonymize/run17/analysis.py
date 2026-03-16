import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Column names
skin1 = "feature18"
skin2 = "feature19"
red = "feature16"
games = "feature9"
player_id = "feature1"

# Compute player-level skin ratings to avoid dyad-level missingness artifacts
player_skin = (
    df[[player_id, skin1, skin2]]
    .groupby(player_id, as_index=False)
    .agg(skin1=(skin1, "mean"), skin2=(skin2, "mean"))
)

# Combine raters at player level
player_skin["skin"] = player_skin[["skin1", "skin2"]].mean(axis=1, skipna=True)

# Aggregate outcomes to player level
player_outcomes = (
    df[[player_id, red, games]]
    .groupby(player_id, as_index=False)
    .agg(red_cards=(red, "sum"), games=(games, "sum"))
)

player = player_outcomes.merge(player_skin[[player_id, "skin"]], on=player_id, how="left")

# Filter valid players with skin rating and exposure
player = player[player["games"] > 0].copy()
player = player[player["skin"].notna()].copy()
player["rate"] = player["red_cards"] / player["games"]

# Poisson regression with offset log(games)
X = sm.add_constant(player["skin"])
model = sm.GLM(player["red_cards"], X, family=sm.families.Poisson(), offset=np.log(player["games"]))
res = model.fit()

beta = res.params["skin"]
se = res.bse["skin"]
pval = res.pvalues["skin"]
rate_ratio = np.exp(beta)

# 95% CI for rate ratio
ci_low = np.exp(beta - 1.96 * se)
ci_high = np.exp(beta + 1.96 * se)

# Binary comparison: light (<=0.25) vs dark (>=0.75)
player["group"] = np.where(player["skin"] >= 0.75, "dark", np.where(player["skin"] <= 0.25, "light", "mid"))

bin_df = player[player["group"].isin(["light", "dark"])].copy()

bin_summary = (
    bin_df
    .groupby("group", as_index=False)
    .agg(red_cards=("red_cards", "sum"), games=("games", "sum"), players=("skin", "count"))
)

# Poisson regression on binary group (dark vs light)
if bin_df["group"].nunique() == 2:
    bin_df["dark"] = (bin_df["group"] == "dark").astype(int)
    Xb = sm.add_constant(bin_df["dark"])
    model_b = sm.GLM(bin_df["red_cards"], Xb, family=sm.families.Poisson(), offset=np.log(bin_df["games"]))
    res_b = model_b.fit()

    beta_b = res_b.params["dark"]
    se_b = res_b.bse["dark"]
    pval_b = res_b.pvalues["dark"]
    rr_b = np.exp(beta_b)
    ci_low_b = np.exp(beta_b - 1.96 * se_b)
    ci_high_b = np.exp(beta_b + 1.96 * se_b)
else:
    beta_b = se_b = pval_b = rr_b = ci_low_b = ci_high_b = np.nan

# Correlation between skin and rate (Spearman)
rho, p_rho = stats.spearmanr(player["skin"], player["rate"])

results = {
    "n_players": int(player.shape[0]),
    "n_players_bin": int(bin_df.shape[0]),
    "poisson_continuous": {
        "beta_skin": float(beta),
        "se": float(se),
        "p_value": float(pval),
        "rate_ratio": float(rate_ratio),
        "rr_ci_low": float(ci_low),
        "rr_ci_high": float(ci_high),
    },
    "poisson_dark_vs_light": {
        "beta_dark": None if np.isnan(beta_b) else float(beta_b),
        "se": None if np.isnan(se_b) else float(se_b),
        "p_value": None if np.isnan(pval_b) else float(pval_b),
        "rate_ratio": None if np.isnan(rr_b) else float(rr_b),
        "rr_ci_low": None if np.isnan(ci_low_b) else float(ci_low_b),
        "rr_ci_high": None if np.isnan(ci_high_b) else float(ci_high_b),
    },
    "group_summary": bin_summary.to_dict(orient="records"),
    "spearman": {"rho": float(rho), "p_value": float(p_rho)},
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
