import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "soccer.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone rating: mean of two raters when available
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

    # Remove rows without skin tone or games
    df = df[df["skin_tone"].notna() & df["games"].notna()]

    # Dyad-level Poisson regression with exposure offset
    # redCards ~ skin_tone + offset(log(games))
    # cluster-robust SE by player to handle repeated measures
    dyad = df.copy()
    dyad = dyad[dyad["games"] > 0]
    dyad["log_games"] = np.log(dyad["games"])

    X = sm.add_constant(dyad[["skin_tone"]])
    y = dyad["redCards"]
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=dyad["log_games"])
    res = model.fit(cov_type="cluster", cov_kwds={"groups": dyad["playerShort"]})

    coef = res.params["skin_tone"]
    pval = res.pvalues["skin_tone"]
    rr = np.exp(coef)
    ci_low, ci_high = np.exp(res.conf_int().loc["skin_tone"].values)

    # Predicted rate ratio for dark vs light
    # Use light=0.25, dark=0.75 (two steps apart, delta=0.5)
    delta = 0.5
    rr_dark_vs_light = np.exp(coef * delta)

    # Player-level aggregation for intuitive rates
    agg = (
        df.groupby("playerShort")
        .agg(
            skin_tone=("skin_tone", "first"),
            total_games=("games", "sum"),
            total_red=("redCards", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["total_games"] > 0]
    agg["rate_red_per_game"] = agg["total_red"] / agg["total_games"]

    # Define light and dark groups (<=0.25 and >=0.75)
    light = agg[agg["skin_tone"] <= 0.25]
    dark = agg[agg["skin_tone"] >= 0.75]

    # Poisson rate comparison between groups at player level
    # Fit Poisson with offset log(games) and binary dark indicator
    agg["dark_group"] = (agg["skin_tone"] >= 0.75).astype(int)
    agg = agg[agg["skin_tone"].isin([0.0, 0.25, 0.5, 0.75, 1.0])]

    Xg = sm.add_constant(agg[["dark_group"]])
    yg = agg["total_red"]
    model_g = sm.GLM(yg, Xg, family=sm.families.Poisson(), offset=np.log(agg["total_games"]))
    res_g = model_g.fit()

    coef_g = res_g.params["dark_group"]
    pval_g = res_g.pvalues["dark_group"]
    rr_g = np.exp(coef_g)
    ci_g_low, ci_g_high = np.exp(res_g.conf_int().loc["dark_group"].values)

    # Descriptive rates
    light_rate = light["total_red"].sum() / light["total_games"].sum() if len(light) else np.nan
    dark_rate = dark["total_red"].sum() / dark["total_games"].sum() if len(dark) else np.nan

    # Two-sample test of rates using Poisson exact (approx via rate ratio test)
    # Compute rate ratio and Wald CI
    rate_ratio = (dark_rate / light_rate) if light_rate and dark_rate else np.nan

    # Also compute correlation at player level
    corr, corr_p = stats.spearmanr(agg["skin_tone"], agg["rate_red_per_game"], nan_policy="omit")

    results = {
        "dyad_poisson": {
            "coef_skin_tone": float(coef),
            "p_value": float(pval),
            "rate_ratio_per_1unit": float(rr),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "rr_dark_vs_light_delta_0_5": float(rr_dark_vs_light),
            "n_rows": int(len(dyad)),
            "n_players": int(dyad["playerShort"].nunique()),
        },
        "player_groups": {
            "n_light": int(len(light)),
            "n_dark": int(len(dark)),
            "light_rate": float(light_rate) if not np.isnan(light_rate) else None,
            "dark_rate": float(dark_rate) if not np.isnan(dark_rate) else None,
            "rate_ratio": float(rate_ratio) if not np.isnan(rate_ratio) else None,
            "poisson_rr_dark": float(rr_g),
            "poisson_p": float(pval_g),
            "poisson_ci_low": float(ci_g_low),
            "poisson_ci_high": float(ci_g_high),
        },
        "spearman": {
            "rho": float(corr),
            "p_value": float(corr_p),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
