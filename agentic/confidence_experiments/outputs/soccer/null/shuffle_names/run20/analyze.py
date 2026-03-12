import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Map shuffled columns using descriptions from info.json
# rater1: skin rating (normalized 0-1) by rater 1
# nExp: skin rating (normalized 0-1) by rater 2
# yellowCards: number of red cards player received from referee
# redCards: number of games in the player-referee dyad (exposure)

skin = df[["rater1", "nExp"]].mean(axis=1)
red_cards = df["yellowCards"]
games = df["redCards"]

# Basic cleaning
mask = skin.notna() & red_cards.notna() & games.notna() & (games > 0)
work = df.loc[mask].copy()
work["skin"] = skin[mask]
work["red_cards"] = red_cards[mask]
work["games"] = games[mask]
work["rate"] = work["red_cards"] / work["games"]

# Define light vs dark based on extreme categories of 5-point scale (0, 0.25, 0.5, 0.75, 1)
work["skin_group"] = np.where(work["skin"] <= 0.25, "light",
                        np.where(work["skin"] >= 0.75, "dark", "mid"))

# Descriptive stats by group
summary = work.groupby("skin_group").agg(
    n=("skin", "size"),
    mean_skin=("skin", "mean"),
    mean_red_cards=("red_cards", "mean"),
    mean_games=("games", "mean"),
    mean_rate=("rate", "mean")
).reset_index()

# Poisson regression with log(games) offset for exposure
X = sm.add_constant(work["skin"])
model = sm.GLM(
    work["red_cards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(work["games"])
).fit(cov_type="HC1")

beta = model.params["skin"]
se = model.bse["skin"]
pval = model.pvalues["skin"]
irr = float(np.exp(beta))

# Poisson with binary dark vs light (exclude mid)
work_bin = work[work["skin_group"].isin(["light", "dark"])].copy()
work_bin["dark"] = (work_bin["skin_group"] == "dark").astype(int)
Xb = sm.add_constant(work_bin["dark"])
model_bin = sm.GLM(
    work_bin["red_cards"],
    Xb,
    family=sm.families.Poisson(),
    offset=np.log(work_bin["games"])
).fit(cov_type="HC1")

beta_bin = model_bin.params["dark"]
se_bin = model_bin.bse["dark"]
pval_bin = model_bin.pvalues["dark"]
irr_bin = float(np.exp(beta_bin))

# Logistic regression for any red card (robust check)
work["any_red"] = (work["red_cards"] > 0).astype(int)
Xl = sm.add_constant(work["skin"])
logit = sm.GLM(
    work["any_red"],
    Xl,
    family=sm.families.Binomial()
).fit(cov_type="HC1")

beta_logit = logit.params["skin"]
se_logit = logit.bse["skin"]
pval_logit = logit.pvalues["skin"]
odds_ratio = float(np.exp(beta_logit))

# Save key results to json for later use
results = {
    "n_rows": int(len(work)),
    "summary": summary.to_dict(orient="records"),
    "poisson_skin": {
        "beta": float(beta),
        "se": float(se),
        "pval": float(pval),
        "irr": float(irr)
    },
    "poisson_dark_vs_light": {
        "beta": float(beta_bin),
        "se": float(se_bin),
        "pval": float(pval_bin),
        "irr": float(irr_bin),
        "n_rows": int(len(work_bin))
    },
    "logit_any_red": {
        "beta": float(beta_logit),
        "se": float(se_logit),
        "pval": float(pval_logit),
        "odds_ratio": float(odds_ratio)
    }
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
