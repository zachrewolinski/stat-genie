import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Compute average skin tone (0-1 scale)
_df["skin_avg"] = _df[["feature18", "feature19"]].mean(axis=1)

# Keep rows with skin tone and positive games
_df = _df.dropna(subset=["skin_avg", "feature9", "feature16"]).copy()
_df = _df[_df["feature9"] > 0].copy()

# Helper: Poisson regression with offset for exposure

def poisson_with_offset(y, x, offset, clusters=None):
    x = sm.add_constant(x, has_constant='add')
    model = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset)
    if clusters is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    else:
        res = model.fit(cov_type="HC0")
    return res

# Continuous skin tone model
_y = _df["feature16"].astype(float)
_offset = np.log(_df["feature9"].astype(float))
_x = _df[["skin_avg"]].astype(float)
_clusters = _df["feature1"].astype(str)  # player short name
res_cont = poisson_with_offset(_y, _x, _offset, clusters=_clusters)

# Binary dark vs light
_df["skin_group"] = pd.Series(np.select(
    [_df["skin_avg"] <= 0.25, _df["skin_avg"] >= 0.75],
    ["light", "dark"],
    default="mid"
), index=_df.index)

_df_dl = _df[_df["skin_group"].isin(["light", "dark"])].copy()
_df_dl["dark"] = (_df_dl["skin_group"] == "dark").astype(int)

_y_dl = _df_dl["feature16"].astype(float)
_offset_dl = np.log(_df_dl["feature9"].astype(float))
_x_dl = _df_dl[["dark"]].astype(float)
_clusters_dl = _df_dl["feature1"].astype(str)
res_dl = poisson_with_offset(_y_dl, _x_dl, _offset_dl, clusters=_clusters_dl)

# Summary stats for rates
rate_light = (_df_dl.loc[_df_dl["dark"] == 0, "feature16"].sum() /
              _df_dl.loc[_df_dl["dark"] == 0, "feature9"].sum())
rate_dark = (_df_dl.loc[_df_dl["dark"] == 1, "feature16"].sum() /
             _df_dl.loc[_df_dl["dark"] == 1, "feature9"].sum())
rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = res_cont.pearson_chi2
pearson_df = res_cont.df_resid
overdisp = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

out = {
    "n_rows": int(len(_df)),
    "n_players": int(_df["feature1"].nunique()),
    "n_rows_dark_light": int(len(_df_dl)),
    "rate_light": float(rate_light),
    "rate_dark": float(rate_dark),
    "rate_ratio_dark_light": float(rate_ratio),
    "cont_beta": float(res_cont.params["skin_avg"]),
    "cont_se": float(res_cont.bse["skin_avg"]),
    "cont_p": float(res_cont.pvalues["skin_avg"]),
    "cont_irr": float(np.exp(res_cont.params["skin_avg"])),
    "dl_beta": float(res_dl.params["dark"]),
    "dl_se": float(res_dl.bse["dark"]),
    "dl_p": float(res_dl.pvalues["dark"]),
    "dl_irr": float(np.exp(res_dl.params["dark"])),
    "overdispersion": float(overdisp),
}

# Print as JSON
import json
print(json.dumps(out, indent=2))
