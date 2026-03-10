import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Ensure numeric columns
for col in ["rater1", "rater2", "redCards", "games"]:
    _df[col] = pd.to_numeric(_df[col], errors="coerce")

# Compute average skin tone
_df["skin_avg"] = _df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin ratings and games
_df = _df.dropna(subset=["skin_avg", "redCards", "games"])

# Avoid zero games for offset
_df = _df[_df["games"] > 0].copy()

# Binary extremes: light <= 0.25, dark >= 0.75
_df["skin_group"] = np.where(_df["skin_avg"] <= 0.25, "light",
                             np.where(_df["skin_avg"] >= 0.75, "dark", "mid"))

# Subset extremes
_ext = _df[_df["skin_group"].isin(["light", "dark"])].copy()

# Aggregate descriptive stats
rate_all = _df["redCards"].sum() / _df["games"].sum()

# Group stats for extremes
_grp = _ext.groupby("skin_group").agg(
    redCards_sum=("redCards", "sum"),
    games_sum=("games", "sum"),
    dyads=("redCards", "size")
)
_grp["rate_per_game"] = _grp["redCards_sum"] / _grp["games_sum"]

# Poisson regression on continuous skin tone with log(games) offset
_df["log_games"] = np.log(_df["games"])

poisson_cont = smf.glm(
    formula="redCards ~ skin_avg",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_games"],
).fit(cov_type="HC1")

# Poisson regression on extreme groups: dark vs light
_ext = _ext.copy()
_ext["log_games"] = np.log(_ext["games"])
_ext["dark"] = (_ext["skin_group"] == "dark").astype(int)

poisson_bin = smf.glm(
    formula="redCards ~ dark",
    data=_ext,
    family=sm.families.Poisson(),
    offset=_ext["log_games"],
).fit(cov_type="HC1")

# Logistic regression for any red card (extremes), with games as covariate
_ext["any_red"] = (_ext["redCards"] > 0).astype(int)
logit = smf.logit("any_red ~ dark + np.log(games)", data=_ext).fit(disp=False)

results = {
    "n_rows": int(_df.shape[0]),
    "n_rows_ext": int(_ext.shape[0]),
    "overall_rate_per_game": rate_all,
    "group_stats": _grp.reset_index().to_dict(orient="records"),
    "poisson_cont": {
        "coef_skin_avg": poisson_cont.params["skin_avg"],
        "se_skin_avg": poisson_cont.bse["skin_avg"],
        "p_skin_avg": poisson_cont.pvalues["skin_avg"],
        "rate_ratio_skin_avg": float(np.exp(poisson_cont.params["skin_avg"]))
    },
    "poisson_bin": {
        "coef_dark": poisson_bin.params["dark"],
        "se_dark": poisson_bin.bse["dark"],
        "p_dark": poisson_bin.pvalues["dark"],
        "rate_ratio_dark_vs_light": float(np.exp(poisson_bin.params["dark"]))
    },
    "logit": {
        "coef_dark": logit.params["dark"],
        "se_dark": logit.bse["dark"],
        "p_dark": logit.pvalues["dark"],
        "odds_ratio_dark_vs_light": float(np.exp(logit.params["dark"]))
    }
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
