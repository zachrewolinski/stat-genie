import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone
# Some rows may have missing rater values
skin_cols = ["rater1", "rater2"]
df["skin_mean"] = df[skin_cols].mean(axis=1)

# Basic filtering
initial_n = len(df)
df = df.dropna(subset=["skin_mean", "redCards", "games"])
df = df[df["games"] > 0]

# Binary dark/light: dark if above neutral (0.5)
df["dark"] = (df["skin_mean"] > 0.5).astype(int)

# Group rates (dark vs light threshold at 0.5)
grp = df.groupby("dark", observed=True)
summary = grp.agg(
    dyads=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
)
summary["red_per_game"] = summary["total_red"] / summary["total_games"]
summary["red_per_100_games"] = summary["red_per_game"] * 100

# Coarse skin categories for descriptive trend
bins = [-0.01, 0.125, 0.375, 0.625, 0.875, 1.01]
labels = ["very_light", "light", "medium", "dark", "very_dark"]
df["skin_cat"] = pd.cut(df["skin_mean"], bins=bins, labels=labels)
cat_summary = (
    df.groupby("skin_cat", observed=True)
    .agg(dyads=("redCards", "size"), total_red=("redCards", "sum"), total_games=("games", "sum"))
    .assign(red_per_game=lambda d: d["total_red"] / d["total_games"])
    .assign(red_per_100_games=lambda d: d["red_per_game"] * 100)
    .reset_index()
)

# Poisson regression with offset (games)
X = sm.add_constant(df["dark"])
model = sm.GLM(df["redCards"], X, family=sm.families.Poisson(), offset=np.log(df["games"]))
# Cluster-robust SE by player to reduce dependence within player
res = model.fit(cov_type="cluster", cov_kwds={"groups": df["playerShort"]})

# Continuous skin tone model (0-1)
Xc = sm.add_constant(df["skin_mean"])
model_c = sm.GLM(df["redCards"], Xc, family=sm.families.Poisson(), offset=np.log(df["games"]))
res_c = model_c.fit(cov_type="cluster", cov_kwds={"groups": df["playerShort"]})

# Extreme comparison: very_dark vs very_light
extreme = df[df["skin_cat"].isin(["very_light", "very_dark"])].copy()
extreme["very_dark"] = (extreme["skin_cat"] == "very_dark").astype(int)
Xext = sm.add_constant(extreme["very_dark"])
model_ext = sm.GLM(extreme["redCards"], Xext, family=sm.families.Poisson(), offset=np.log(extreme["games"]))
res_ext = model_ext.fit(cov_type="cluster", cov_kwds={"groups": extreme["playerShort"]})

# Overdispersion check
# deviance/df_resid > 1 suggests overdispersion
overdispersion = res.deviance / res.df_resid

# Prepare outputs
out = {
    "initial_rows": int(initial_n),
    "rows_used": int(len(df)),
    "group_summary": summary.reset_index().to_dict(orient="records"),
    "poisson_dark_coef": float(res.params["dark"]),
    "poisson_dark_se": float(res.bse["dark"]),
    "poisson_dark_p": float(res.pvalues["dark"]),
    "poisson_dark_irr": float(np.exp(res.params["dark"])),
    "poisson_dark_ci_low": float(np.exp(res.conf_int().loc["dark"][0])),
    "poisson_dark_ci_high": float(np.exp(res.conf_int().loc["dark"][1])),
    "poisson_skin_coef": float(res_c.params["skin_mean"]),
    "poisson_skin_se": float(res_c.bse["skin_mean"]),
    "poisson_skin_p": float(res_c.pvalues["skin_mean"]),
    "poisson_skin_irr": float(np.exp(res_c.params["skin_mean"])),
    "poisson_skin_ci_low": float(np.exp(res_c.conf_int().loc["skin_mean"][0])),
    "poisson_skin_ci_high": float(np.exp(res_c.conf_int().loc["skin_mean"][1])),
    "extreme_rows_used": int(len(extreme)),
    "extreme_summary": extreme.groupby("very_dark", observed=True).agg(
        dyads=("redCards", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
    ).assign(
        red_per_game=lambda d: d["total_red"] / d["total_games"],
        red_per_100_games=lambda d: d["red_per_game"] * 100,
    ).reset_index().to_dict(orient="records"),
    "poisson_extreme_coef": float(res_ext.params["very_dark"]),
    "poisson_extreme_se": float(res_ext.bse["very_dark"]),
    "poisson_extreme_p": float(res_ext.pvalues["very_dark"]),
    "poisson_extreme_irr": float(np.exp(res_ext.params["very_dark"])),
    "poisson_extreme_ci_low": float(np.exp(res_ext.conf_int().loc["very_dark"][0])),
    "poisson_extreme_ci_high": float(np.exp(res_ext.conf_int().loc["very_dark"][1])),
    "overdispersion": float(overdispersion),
    "skin_cat_summary": cat_summary.to_dict(orient="records"),
}

pd.set_option("display.max_columns", None)
print(out)
