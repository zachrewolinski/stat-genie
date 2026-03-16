import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

DATA_PATH = "soccer.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Compute skin tone as mean of rater1 and rater2 (when available)
_df["skin_tone"] = _df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Basic cleaning
_df = _df[_df["skin_tone"].notna()]
_df = _df[_df["games"].notna() & (_df["games"] > 0)]
_df = _df[_df["redCards"].notna()]

# Binary groups: light <= 0.25, dark >= 0.75
_df["dark"] = (_df["skin_tone"] >= 0.75).astype(int)
_df["light"] = (_df["skin_tone"] <= 0.25).astype(int)
_df_bin = _df[(_df["dark"] == 1) | (_df["light"] == 1)].copy()
_df_bin["dark"] = (_df_bin["skin_tone"] >= 0.75).astype(int)

# Aggregate rates by group
_group = _df_bin.groupby("dark").agg(
    n_rows=("redCards", "size"),
    total_games=("games", "sum"),
    total_red=("redCards", "sum"),
)
_group["rate_per_game"] = _group["total_red"] / _group["total_games"]

# Poisson GLM with offset
X = sm.add_constant(_df_bin["dark"])
offset = np.log(_df_bin["games"])
poisson_model = sm.GLM(_df_bin["redCards"], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC0")

# Negative Binomial with exposure (games)
nb_model = smd.NegativeBinomial(_df_bin["redCards"], X, exposure=_df_bin["games"])
nb_res = nb_model.fit(disp=False)

# Continuous skin tone model (Poisson)
Xc = sm.add_constant(_df[["skin_tone"]])
offset_c = np.log(_df["games"])
poisson_c = sm.GLM(_df["redCards"], Xc, family=sm.families.Poisson(), offset=offset_c)
poisson_c_res = poisson_c.fit(cov_type="HC0")

# Effect sizes
poisson_rr = float(np.exp(poisson_res.params["dark"]))
poisson_rr_ci = np.exp(poisson_res.conf_int().loc["dark"].values).tolist()
poisson_p = float(poisson_res.pvalues["dark"])

nb_rr = float(np.exp(nb_res.params["dark"]))
nb_rr_ci = np.exp(nb_res.conf_int().loc["dark"].values).tolist()
nb_p = float(nb_res.pvalues["dark"])

beta_c = float(poisson_c_res.params["skin_tone"])
# Compare dark (0.75) vs light (0.25) on continuous scale
rr_dark_light_cont = float(np.exp(beta_c * (0.75 - 0.25)))

results = {
    "n_total_rows": int(len(_df)),
    "n_binary_rows": int(len(_df_bin)),
    "group_rates": {
        "light": {
            "n_rows": int(_group.loc[0, "n_rows"]),
            "total_games": float(_group.loc[0, "total_games"]),
            "total_red": float(_group.loc[0, "total_red"]),
            "rate_per_game": float(_group.loc[0, "rate_per_game"]),
        },
        "dark": {
            "n_rows": int(_group.loc[1, "n_rows"]),
            "total_games": float(_group.loc[1, "total_games"]),
            "total_red": float(_group.loc[1, "total_red"]),
            "rate_per_game": float(_group.loc[1, "rate_per_game"]),
        },
    },
    "poisson_binary": {
        "rate_ratio_dark_vs_light": poisson_rr,
        "ci": poisson_rr_ci,
        "p_value": poisson_p,
        "deviance_over_df": float(poisson_res.deviance / poisson_res.df_resid),
    },
    "neg_bin_binary": {
        "rate_ratio_dark_vs_light": nb_rr,
        "ci": nb_rr_ci,
        "p_value": nb_p,
        "alpha": float(nb_res.params[-1]),
    },
    "poisson_continuous": {
        "beta_skin_tone": beta_c,
        "rate_ratio_dark_vs_light_equivalent": rr_dark_light_cont,
        "p_value": float(poisson_c_res.pvalues["skin_tone"]),
    },
}

print(json.dumps(results, indent=2))
