import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load only needed columns
cols = [
    "playerShort",
    "refNum",
    "games",
    "redCards",
    "rater1",
    "rater2",
]

df = pd.read_csv(DATA_PATH, usecols=cols)

# Basic cleaning
for c in ["games", "redCards", "rater1", "rater2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[(df["games"] > 0) & (df["redCards"].notna())]

df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin ratings
skin_df = df[df["skin_mean"].notna()].copy()

# Define light/dark extremes (0, 0.25 vs 0.75, 1.0)
conditions = [skin_df["skin_mean"] <= 0.25, skin_df["skin_mean"] >= 0.75]
choices = ["light", "dark"]
skin_df["skin_group"] = np.select(conditions, choices, default="mid")

# Summary counts
summary = {
    "n_rows": int(len(df)),
    "n_rows_with_skin": int(len(skin_df)),
    "group_counts": skin_df["skin_group"].value_counts().to_dict(),
}

# Group stats (red cards per game)
skin_df["red_per_game"] = skin_df["redCards"] / skin_df["games"]

group_stats = (
    skin_df.groupby("skin_group")
    .agg(
        n=("redCards", "size"),
        red_cards=("redCards", "sum"),
        games=("games", "sum"),
        mean_red_per_game=("red_per_game", "mean"),
        prop_any_red=("redCards", lambda x: float((x > 0).mean())),
    )
)

group_stats["rate_red_per_game"] = group_stats["red_cards"] / group_stats["games"]

# Filter to light vs dark only
ld = skin_df[skin_df["skin_group"].isin(["light", "dark"])].copy()
ld["dark"] = (ld["skin_group"] == "dark").astype(int)

results = {}

# Poisson regression at dyad level with offset log(games)
try:
    X = sm.add_constant(ld[["dark"]])
    model = sm.GLM(
        ld["redCards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(ld["games"]),
    )
    res = model.fit()
    try:
        res_rob = res.get_robustcov_results(cov_type="HC3")
    except Exception:
        res_rob = res

    coef = float(res_rob.params["dark"])
    se = float(res_rob.bse["dark"])
    p = float(res_rob.pvalues["dark"])
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    results["dyad_poisson"] = {
        "coef_dark": coef,
        "se_dark": se,
        "p_dark": p,
        "rate_ratio_dark_vs_light": rr,
        "rr_95ci": [ci_low, ci_high],
        "n": int(len(ld)),
    }
except Exception as e:
    results["dyad_poisson_error"] = str(e)

# Player-level aggregation to reduce dependence
player = (
    skin_df.groupby(["playerShort", "skin_group"], as_index=False)
    .agg(redCards=("redCards", "sum"), games=("games", "sum"))
)

player_ld = player[player["skin_group"].isin(["light", "dark"])].copy()
player_ld["dark"] = (player_ld["skin_group"] == "dark").astype(int)

try:
    Xp = sm.add_constant(player_ld[["dark"]])
    model_p = sm.GLM(
        player_ld["redCards"],
        Xp,
        family=sm.families.Poisson(),
        offset=np.log(player_ld["games"]),
    )
    res_p = model_p.fit()
    try:
        res_p_rob = res_p.get_robustcov_results(cov_type="HC3")
    except Exception:
        res_p_rob = res_p

    coef_p = float(res_p_rob.params["dark"])
    se_p = float(res_p_rob.bse["dark"])
    p_p = float(res_p_rob.pvalues["dark"])
    rr_p = float(np.exp(coef_p))
    ci_low_p = float(np.exp(coef_p - 1.96 * se_p))
    ci_high_p = float(np.exp(coef_p + 1.96 * se_p))

    results["player_poisson"] = {
        "coef_dark": coef_p,
        "se_dark": se_p,
        "p_dark": p_p,
        "rate_ratio_dark_vs_light": rr_p,
        "rr_95ci": [ci_low_p, ci_high_p],
        "n_players": int(len(player_ld)),
    }
except Exception as e:
    results["player_poisson_error"] = str(e)

# Save outputs for inspection
output = {
    "summary": summary,
    "group_stats": group_stats.reset_index().to_dict(orient="records"),
    "results": results,
}

print(json.dumps(output, indent=2))
