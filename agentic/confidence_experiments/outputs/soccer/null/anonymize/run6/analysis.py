import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Column names
COL_PLAYER = "feature1"
COL_GAMES = "feature9"
COL_RED = "feature16"
COL_SKIN1 = "feature18"
COL_SKIN2 = "feature19"

# Compute average skin tone (0-1 scale)
df["skin_avg"] = df[[COL_SKIN1, COL_SKIN2]].mean(axis=1)

# Keep needed columns and drop missing/invalid rows
use_cols = [COL_PLAYER, "skin_avg", COL_GAMES, COL_RED]
df = df[use_cols].dropna()
df = df[df[COL_GAMES] > 0]

# ----------------------
# Player-level aggregation
# ----------------------
player = (
    df.groupby(COL_PLAYER, as_index=False)
      .agg(skin_avg=("skin_avg", "mean"),
           games=(COL_GAMES, "sum"),
           red=(COL_RED, "sum"))
)
player = player[player["games"] > 0]

# Continuous Poisson regression: red ~ skin_avg + offset(log(games))
X_p = sm.add_constant(player["skin_avg"])
model_p = sm.GLM(
    player["red"],
    X_p,
    family=sm.families.Poisson(),
    offset=np.log(player["games"])
)
res_p = model_p.fit(cov_type="HC0")

rr_skin_p = float(np.exp(res_p.params["skin_avg"]))
ci_p = res_p.conf_int().loc["skin_avg"].astype(float)
rr_ci_p = (float(np.exp(ci_p[0])), float(np.exp(ci_p[1])))

# Quantile-based dark vs light (top/bottom 20%)
q20, q80 = player["skin_avg"].quantile([0.2, 0.8])
player["group_q"] = np.where(
    player["skin_avg"] <= q20,
    "light",
    np.where(player["skin_avg"] >= q80, "dark", "mid"),
)
subset_q = player[player["group_q"].isin(["light", "dark"])].copy()
subset_q["dark"] = (subset_q["group_q"] == "dark").astype(int)

X_q = sm.add_constant(subset_q["dark"])
model_q = sm.GLM(
    subset_q["red"],
    X_q,
    family=sm.families.Poisson(),
    offset=np.log(subset_q["games"])
)
res_q = model_q.fit(cov_type="HC0")

rr_dark_q = float(np.exp(res_q.params["dark"]))
ci_q = res_q.conf_int().loc["dark"].astype(float)
rr_dark_ci = (float(np.exp(ci_q[0])), float(np.exp(ci_q[1])))

rate_light_q = float(
    subset_q.loc[subset_q["group_q"] == "light", "red"].sum()
    / subset_q.loc[subset_q["group_q"] == "light", "games"].sum()
)
rate_dark_q = float(
    subset_q.loc[subset_q["group_q"] == "dark", "red"].sum()
    / subset_q.loc[subset_q["group_q"] == "dark", "games"].sum()
)

# ----------------------
# Dyad-level regression with cluster-robust SE by player
# ----------------------
X_d = sm.add_constant(df["skin_avg"])
model_d = sm.GLM(
    df[COL_RED],
    X_d,
    family=sm.families.Poisson(),
    offset=np.log(df[COL_GAMES])
)
res_d = model_d.fit(cov_type="cluster", cov_kwds={"groups": df[COL_PLAYER]})

rr_skin_d = float(np.exp(res_d.params["skin_avg"]))
ci_d = res_d.conf_int().loc["skin_avg"].astype(float)
rr_ci_d = (float(np.exp(ci_d[0])), float(np.exp(ci_d[1])))

summary = {
    "n_dyads": int(len(df)),
    "n_players": int(len(player)),
    "q20_skin": float(q20),
    "q80_skin": float(q80),
    "n_players_light_q": int((player["group_q"] == "light").sum()),
    "n_players_dark_q": int((player["group_q"] == "dark").sum()),
    "rate_light_q": rate_light_q,
    "rate_dark_q": rate_dark_q,
    "poisson_player_coef": float(res_p.params["skin_avg"]),
    "poisson_player_p": float(res_p.pvalues["skin_avg"]),
    "poisson_player_rr": rr_skin_p,
    "poisson_player_rr_ci": rr_ci_p,
    "poisson_q_coef": float(res_q.params["dark"]),
    "poisson_q_p": float(res_q.pvalues["dark"]),
    "poisson_q_rr": rr_dark_q,
    "poisson_q_rr_ci": rr_dark_ci,
    "poisson_dyad_coef": float(res_d.params["skin_avg"]),
    "poisson_dyad_p": float(res_d.pvalues["skin_avg"]),
    "poisson_dyad_rr": rr_skin_d,
    "poisson_dyad_rr_ci": rr_ci_d,
}

print(pd.Series(summary))
