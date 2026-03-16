import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)

# Drop missing essential values
use_cols = ["playerShort", "games", "redCards", "skin_tone"]
df_use = df[use_cols].dropna()

# Aggregate to player level
player = (
    df_use.groupby("playerShort", as_index=False)
    .agg(
        total_redCards=("redCards", "sum"),
        total_games=("games", "sum"),
        skin_tone=("skin_tone", "mean"),
    )
)

# Avoid zero games just in case
player = player[player["total_games"] > 0].copy()
player["rate"] = player["total_redCards"] / player["total_games"]

# Define dark vs light threshold at 0.5
player["dark"] = player["skin_tone"] >= 0.5

# Group stats
summary = player.groupby("dark").agg(
    n_players=("playerShort", "count"),
    total_redCards=("total_redCards", "sum"),
    total_games=("total_games", "sum"),
    mean_rate=("rate", "mean"),
    median_rate=("rate", "median"),
)
summary["rate_overall"] = summary["total_redCards"] / summary["total_games"]

# Welch t-test on player-level rates
rates_dark = player.loc[player["dark"], "rate"]
rates_light = player.loc[~player["dark"], "rate"]

ttest_res = stats.ttest_ind(rates_dark, rates_light, equal_var=False, nan_policy="omit")

# Mann-Whitney U (non-param)
try:
    mwu_res = stats.mannwhitneyu(rates_dark, rates_light, alternative="two-sided")
except Exception:
    mwu_res = None

# Poisson regression with offset log(games)
# model: total_redCards ~ skin_tone
X = sm.add_constant(player["skin_tone"])
offset = np.log(player["total_games"])
poisson_model = sm.GLM(player["total_redCards"], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Logistic regression for any red card (>=1)
player["any_red"] = (player["total_redCards"] > 0).astype(int)
logit_model = sm.Logit(player["any_red"], X)
logit_res = logit_model.fit(disp=0)

# Spearman correlation between skin tone and rate
spearman = stats.spearmanr(player["skin_tone"], player["rate"], nan_policy="omit")

result = {
    "n_player_level": int(player.shape[0]),
    "summary_by_dark": summary.reset_index().to_dict(orient="records"),
    "ttest_rate": {
        "statistic": float(ttest_res.statistic),
        "pvalue": float(ttest_res.pvalue),
    },
    "mannwhitneyu_rate": None if mwu_res is None else {
        "statistic": float(mwu_res.statistic),
        "pvalue": float(mwu_res.pvalue),
    },
    "poisson_skin": {
        "coef": float(poisson_res.params["skin_tone"]),
        "se": float(poisson_res.bse["skin_tone"]),
        "pvalue": float(poisson_res.pvalues["skin_tone"]),
        "exp_coef": float(np.exp(poisson_res.params["skin_tone"])),
    },
    "logit_any_red": {
        "coef": float(logit_res.params["skin_tone"]),
        "se": float(logit_res.bse["skin_tone"]),
        "pvalue": float(logit_res.pvalues["skin_tone"]),
        "odds_ratio": float(np.exp(logit_res.params["skin_tone"])),
    },
    "spearman": {
        "rho": float(spearman.correlation),
        "pvalue": float(spearman.pvalue),
    },
}

print(json.dumps(result, indent=2))
