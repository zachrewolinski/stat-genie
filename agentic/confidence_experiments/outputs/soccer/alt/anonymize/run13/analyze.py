import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Core variables
_df["skin_mean"] = _df[["feature18", "feature19"]].mean(axis=1)
_df = _df.dropna(subset=["skin_mean", "feature16", "feature9", "feature1"])
_df = _df[_df["feature9"] > 0]

# Descriptive by skin_mean category values
skin_levels = sorted(_df["skin_mean"].dropna().unique())
level_summary = []
for lvl in skin_levels:
    sub = _df[_df["skin_mean"] == lvl]
    total_red = sub["feature16"].sum()
    total_games = sub["feature9"].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    level_summary.append({
        "skin_mean": float(lvl),
        "n_dyads": int(len(sub)),
        "total_red": float(total_red),
        "total_games": float(total_games),
        "red_per_game": float(rate),
    })

# Binary extremes: light (<=0.25) vs dark (>=0.75)
_df["skin_cat"] = np.where(_df["skin_mean"] <= 0.25, "light",
                     np.where(_df["skin_mean"] >= 0.75, "dark", "mid"))
_df_bin = _df[_df["skin_cat"].isin(["light", "dark"])].copy()

# Poisson GLM with offset and cluster-robust SEs by player
_df_bin["dark"] = (_df_bin["skin_cat"] == "dark").astype(int)
X_bin = sm.add_constant(_df_bin["dark"])
model_bin = sm.GLM(_df_bin["feature16"], X_bin, family=sm.families.Poisson(), offset=np.log(_df_bin["feature9"]))
res_bin = model_bin.fit(cov_type="cluster", cov_kwds={"groups": _df_bin["feature1"]})

rr_bin = float(np.exp(res_bin.params["dark"]))
ci_bin = np.exp(res_bin.conf_int().loc["dark"]).tolist()
ci_bin = [float(ci_bin[0]), float(ci_bin[1])]

# Continuous skin_mean model
X_cont = sm.add_constant(_df["skin_mean"])
model_cont = sm.GLM(_df["feature16"], X_cont, family=sm.families.Poisson(), offset=np.log(_df["feature9"]))
res_cont = model_cont.fit(cov_type="cluster", cov_kwds={"groups": _df["feature1"]})

beta = float(res_cont.params["skin_mean"])
# Effect from 0.25 to 0.75 (one step from light to dark-ish)
rr_025_075 = float(np.exp(beta * (0.75 - 0.25)))
ci_cont = np.exp(res_cont.conf_int().loc["skin_mean"]).tolist()
ci_cont = [float(ci_cont[0]), float(ci_cont[1])]

# Descriptive rates for binary
def rate_stats(df):
    total_red = df["feature16"].sum()
    total_games = df["feature9"].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    return float(total_red), float(total_games), float(rate)

light_stats = rate_stats(_df_bin[_df_bin["skin_cat"] == "light"])
dark_stats = rate_stats(_df_bin[_df_bin["skin_cat"] == "dark"])

output = {
    "n_rows_used": int(len(_df)),
    "n_rows_bin": int(len(_df_bin)),
    "level_summary": level_summary,
    "bin": {
        "rr_dark_vs_light": rr_bin,
        "ci_dark_vs_light": ci_bin,
        "pvalue_dark": float(res_bin.pvalues["dark"]),
        "light_total_red": light_stats[0],
        "light_total_games": light_stats[1],
        "light_red_per_game": light_stats[2],
        "dark_total_red": dark_stats[0],
        "dark_total_games": dark_stats[1],
        "dark_red_per_game": dark_stats[2],
    },
    "cont": {
        "beta_skin_mean": beta,
        "ci_beta": ci_cont,
        "pvalue_skin_mean": float(res_cont.pvalues["skin_mean"]),
        "rr_0.25_to_0.75": rr_025_075,
    },
}

print(json.dumps(output, indent=2))
