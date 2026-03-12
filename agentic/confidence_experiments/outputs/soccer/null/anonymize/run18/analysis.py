import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute average skin tone rating across the two raters
skin_cols = ["feature18", "feature19"]
df["skin_tone"] = df[skin_cols].mean(axis=1, skipna=True)

# Aggregate to player level to avoid repeated dyads per player
player = (
    df.groupby("feature1", as_index=False)
    .agg(
        skin_tone=("skin_tone", "mean"),
        red_cards=("feature16", "sum"),
        games=("feature9", "sum"),
    )
)

# Define skin tone groups
player["skin_group"] = np.where(
    player["skin_tone"] > 0.5,
    "dark",
    np.where(player["skin_tone"] < 0.5, "light", "neutral"),
)

# Filter to players with valid skin tone and positive exposure
player = player[~player["skin_tone"].isna() & (player["games"] > 0)]

# Primary analysis: compare dark vs light only (exclude neutral)
ld = player[player["skin_group"].isin(["light", "dark"])].copy()
ld["dark"] = (ld["skin_group"] == "dark").astype(int)
ld["log_games"] = np.log(ld["games"])

summary = (
    ld.groupby("skin_group")
    .apply(
        lambda x: pd.Series(
            {
                "players": len(x),
                "total_red_cards": x["red_cards"].sum(),
                "total_games": x["games"].sum(),
                "red_cards_per_game": x["red_cards"].sum() / x["games"].sum(),
            }
        )
    )
    .reset_index()
)

model = sm.GLM(
    ld["red_cards"],
    sm.add_constant(ld["dark"]),
    family=sm.families.Poisson(),
    offset=ld["log_games"],
)
res = model.fit()

coef = res.params["dark"]
rr = float(np.exp(coef))
ci = res.conf_int().loc["dark"].astype(float)
rr_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
pval = float(res.pvalues["dark"])

# Overdispersion check (deviance / df)
overdisp = float(res.deviance / res.df_resid)

# Sensitivity: treat neutral as light
player_sens = player.copy()
player_sens["skin_group_sens"] = np.where(
    player_sens["skin_tone"] >= 0.5, "dark", "light"
)
ld2 = player_sens.copy()
ld2["dark"] = (ld2["skin_group_sens"] == "dark").astype(int)
ld2["log_games"] = np.log(ld2["games"])

model2 = sm.GLM(
    ld2["red_cards"],
    sm.add_constant(ld2["dark"]),
    family=sm.families.Poisson(),
    offset=ld2["log_games"],
)
res2 = model2.fit()

coef2 = res2.params["dark"]
rr2 = float(np.exp(coef2))
ci2 = res2.conf_int().loc["dark"].astype(float)
rr2_ci = (float(np.exp(ci2[0])), float(np.exp(ci2[1])))
pval2 = float(res2.pvalues["dark"])

# Save results for inspection
results = {
    "summary": summary,
    "rr": rr,
    "rr_ci": rr_ci,
    "pval": pval,
    "overdisp": overdisp,
    "rr_sens": rr2,
    "rr_sens_ci": rr2_ci,
    "pval_sens": pval2,
    "n_players_ld": int(len(ld)),
    "n_players_total": int(len(player)),
}

print(results)
