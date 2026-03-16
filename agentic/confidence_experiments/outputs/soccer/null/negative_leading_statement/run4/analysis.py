import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "soccer.csv"


def rate_ratio_ci(count1, exposure1, count2, exposure2, alpha=0.05):
    # log rate ratio CI using normal approximation; requires counts > 0
    rr = (count1 / exposure1) / (count2 / exposure2)
    se = np.sqrt(1 / count1 + 1 / count2)
    z = stats.norm.ppf(1 - alpha / 2)
    lcl = np.exp(np.log(rr) - z * se)
    ucl = np.exp(np.log(rr) + z * se)
    return rr, lcl, ucl


def main():
    df = pd.read_csv(DATA_PATH)
    # compute skin tone mean
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

    # drop missing skin tone or missing games
    df = df.dropna(subset=["skin_mean", "games", "redCards"])
    df = df[df["games"] > 0]

    # aggregate to player level
    player = (
        df.groupby("playerShort", as_index=False)
        .agg(
            skin_mean=("skin_mean", "mean"),
            total_red=("redCards", "sum"),
            total_games=("games", "sum"),
        )
    )

    # define light/dark groups on 5-point scale mapped to [0,1]
    # 0.0, 0.25, 0.5, 0.75, 1.0
    light_strict = player[player["skin_mean"] <= 0.25].copy()
    dark_strict = player[player["skin_mean"] >= 0.75].copy()

    summary = {}
    summary["n_players_total"] = int(player.shape[0])
    summary["n_players_light_strict"] = int(light_strict.shape[0])
    summary["n_players_dark_strict"] = int(dark_strict.shape[0])

    # rates per game
    def group_stats(g):
        total_red = g["total_red"].sum()
        total_games = g["total_games"].sum()
        rate = total_red / total_games if total_games > 0 else np.nan
        return total_red, total_games, rate

    light_red, light_games, light_rate = group_stats(light_strict)
    dark_red, dark_games, dark_rate = group_stats(dark_strict)

    summary["light_strict_total_red"] = float(light_red)
    summary["light_strict_total_games"] = float(light_games)
    summary["light_strict_red_per_game"] = float(light_rate)
    summary["dark_strict_total_red"] = float(dark_red)
    summary["dark_strict_total_games"] = float(dark_games)
    summary["dark_strict_red_per_game"] = float(dark_rate)

    if light_red > 0 and dark_red > 0:
        rr, rr_lcl, rr_ucl = rate_ratio_ci(dark_red, dark_games, light_red, light_games)
        summary["rate_ratio_dark_vs_light_strict"] = float(rr)
        summary["rate_ratio_strict_ci_lower"] = float(rr_lcl)
        summary["rate_ratio_strict_ci_upper"] = float(rr_ucl)
    else:
        summary["rate_ratio_dark_vs_light_strict"] = None
        summary["rate_ratio_strict_ci_lower"] = None
        summary["rate_ratio_strict_ci_upper"] = None

    # quartile-based light/dark groups to ensure adequate sample sizes
    q25 = player["skin_mean"].quantile(0.25)
    q75 = player["skin_mean"].quantile(0.75)
    light_q = player[player["skin_mean"] <= q25].copy()
    dark_q = player[player["skin_mean"] >= q75].copy()

    summary["skin_mean_q25"] = float(q25)
    summary["skin_mean_q75"] = float(q75)
    summary["n_players_light_quartile"] = int(light_q.shape[0])
    summary["n_players_dark_quartile"] = int(dark_q.shape[0])

    light_q_red, light_q_games, light_q_rate = group_stats(light_q)
    dark_q_red, dark_q_games, dark_q_rate = group_stats(dark_q)

    summary["light_quartile_total_red"] = float(light_q_red)
    summary["light_quartile_total_games"] = float(light_q_games)
    summary["light_quartile_red_per_game"] = float(light_q_rate)
    summary["dark_quartile_total_red"] = float(dark_q_red)
    summary["dark_quartile_total_games"] = float(dark_q_games)
    summary["dark_quartile_red_per_game"] = float(dark_q_rate)

    if light_q_red > 0 and dark_q_red > 0:
        rr_q, rr_q_lcl, rr_q_ucl = rate_ratio_ci(dark_q_red, dark_q_games, light_q_red, light_q_games)
        summary["rate_ratio_dark_vs_light_quartile"] = float(rr_q)
        summary["rate_ratio_quartile_ci_lower"] = float(rr_q_lcl)
        summary["rate_ratio_quartile_ci_upper"] = float(rr_q_ucl)
    else:
        summary["rate_ratio_dark_vs_light_quartile"] = None
        summary["rate_ratio_quartile_ci_lower"] = None
        summary["rate_ratio_quartile_ci_upper"] = None

    # Poisson regression at player level with offset
    player_model = player.copy()
    player_model["intercept"] = 1.0
    offset = np.log(player_model["total_games"])
    glm = sm.GLM(
        player_model["total_red"],
        player_model[["intercept", "skin_mean"]],
        family=sm.families.Poisson(),
        offset=offset,
    )
    res = glm.fit(cov_type="HC0")

    summary["glm_coef_skin_mean"] = float(res.params["skin_mean"])
    summary["glm_se_skin_mean"] = float(res.bse["skin_mean"])
    summary["glm_pvalue_skin_mean"] = float(res.pvalues["skin_mean"])
    summary["glm_rate_ratio_skin_mean"] = float(np.exp(res.params["skin_mean"]))

    # also fit dyad-level GEE with clustering by player
    try:
        gee_df = df.copy()
        gee_df["intercept"] = 1.0
        gee_offset = np.log(gee_df["games"])
        gee_model = sm.GEE(
            gee_df["redCards"],
            gee_df[["intercept", "skin_mean"]],
            groups=gee_df["playerShort"],
            family=sm.families.Poisson(),
            offset=gee_offset,
        )
        gee_res = gee_model.fit()
        summary["gee_coef_skin_mean"] = float(gee_res.params["skin_mean"])
        summary["gee_se_skin_mean"] = float(gee_res.bse["skin_mean"])
        summary["gee_pvalue_skin_mean"] = float(gee_res.pvalues["skin_mean"])
        summary["gee_rate_ratio_skin_mean"] = float(np.exp(gee_res.params["skin_mean"]))
    except Exception as e:
        summary["gee_error"] = str(e)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
