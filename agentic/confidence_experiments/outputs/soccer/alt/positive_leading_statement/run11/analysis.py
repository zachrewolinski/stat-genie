import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Derive skin tone as mean of two raters
rater_cols = ["rater1", "rater2"]
df["skin_tone"] = df[rater_cols].mean(axis=1)

# Keep relevant rows
needed = ["redCards", "games", "skin_tone", "playerShort"]
df = df.dropna(subset=needed)
df = df[df["games"] > 0].copy()


def robust_results(res, **kwargs):
    # statsmodels version compatibility
    if hasattr(res, "get_robustcov_results"):
        return res.get_robustcov_results(**kwargs)
    if hasattr(res, "_get_robustcov_results"):
        return res._get_robustcov_results(**kwargs)
    return res


# Binary grouping: light (<=0.25) vs dark (>=0.75)
df["tone_group"] = np.where(
    df["skin_tone"] <= 0.25,
    "light",
    np.where(df["skin_tone"] >= 0.75, "dark", "mid"),
)

# Continuous model (full data)
X_cont = sm.add_constant(df["skin_tone"])
model_cont = sm.GLM(
    df["redCards"],
    X_cont,
    family=sm.families.Poisson(),
    offset=np.log(df["games"]),
)
res_cont = model_cont.fit()
res_cont_cl = robust_results(res_cont, cov_type="cluster", groups=df["playerShort"])
if res_cont_cl is None:
    res_cont_cl = res_cont

beta_cont = float(res_cont_cl.params["skin_tone"])
p_cont = float(res_cont_cl.pvalues["skin_tone"])
rr_cont = float(np.exp(beta_cont))
ci_cont = res_cont_cl.conf_int().loc["skin_tone"].to_list()
rr_cont_ci = [float(np.exp(ci_cont[0])), float(np.exp(ci_cont[1]))]

# Binary model (dark vs light only)
df_bin = df[df["tone_group"].isin(["light", "dark"])].copy()
df_bin["dark"] = (df_bin["tone_group"] == "dark").astype(int)

X_bin = sm.add_constant(df_bin["dark"])
model_bin = sm.GLM(
    df_bin["redCards"],
    X_bin,
    family=sm.families.Poisson(),
    offset=np.log(df_bin["games"]),
)
res_bin = model_bin.fit()
res_bin_cl = robust_results(res_bin, cov_type="cluster", groups=df_bin["playerShort"])
if res_bin_cl is None:
    res_bin_cl = res_bin

beta_bin = float(res_bin_cl.params["dark"])
p_bin = float(res_bin_cl.pvalues["dark"])
rr_bin = float(np.exp(beta_bin))
ci_bin = res_bin_cl.conf_int().loc["dark"].to_list()
rr_bin_ci = [float(np.exp(ci_bin[0])), float(np.exp(ci_bin[1]))]

# Rate calculations
rate_dark = df_bin.loc[df_bin["dark"] == 1, "redCards"].sum() / df_bin.loc[
    df_bin["dark"] == 1, "games"
].sum()
rate_light = df_bin.loc[df_bin["dark"] == 0, "redCards"].sum() / df_bin.loc[
    df_bin["dark"] == 0, "games"
].sum()
rate_ratio = float(rate_dark / rate_light)

results = {
    "n_rows_total": int(df.shape[0]),
    "n_rows_bin": int(df_bin.shape[0]),
    "n_players_bin": int(df_bin["playerShort"].nunique()),
    "rate_light": float(rate_light),
    "rate_dark": float(rate_dark),
    "rate_ratio_dark_vs_light": rate_ratio,
    "poisson_continuous": {
        "rr_per_unit_skin_tone": rr_cont,
        "rr_ci": rr_cont_ci,
        "p_value": p_cont,
    },
    "poisson_binary": {
        "rr_dark_vs_light": rr_bin,
        "rr_ci": rr_bin_ci,
        "p_value": p_bin,
    },
}

print(json.dumps(results, indent=2))
