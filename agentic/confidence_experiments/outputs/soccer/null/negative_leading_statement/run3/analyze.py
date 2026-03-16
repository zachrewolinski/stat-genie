import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # compute skin tone as mean of rater1 and rater2 when available
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
    return df


def aggregate_player_level(df: pd.DataFrame) -> pd.DataFrame:
    # Use only rows with skin ratings
    df = df.loc[df["skin_tone"].notna()].copy()
    # Aggregate at player level
    agg = (
        df.groupby("playerShort", as_index=False)
        .agg(
            skin_tone=("skin_tone", "first"),
            total_games=("games", "sum"),
            total_red_cards=("redCards", "sum"),
            total_yellow_cards=("yellowCards", "sum"),
            total_yellow_reds=("yellowReds", "sum"),
        )
    )
    # remove players with zero games (shouldn't happen)
    agg = agg.loc[agg["total_games"] > 0].copy()
    agg["red_per_game"] = agg["total_red_cards"] / agg["total_games"]
    return agg


def poisson_regression(agg: pd.DataFrame):
    # Poisson regression with log(games) offset
    y = agg["total_red_cards"].values
    X = sm.add_constant(agg["skin_tone"].values)
    offset = np.log(agg["total_games"].values)
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    result = model.fit(cov_type="HC1")
    # coefficient for skin_tone
    coef = result.params[1]
    se = result.bse[1]
    p = result.pvalues[1]
    # IRR for full 0-1 range and per 0.25 step
    irr_full = math.exp(coef)
    irr_step = math.exp(coef * 0.25)
    ci_low = math.exp(coef - 1.96 * se)
    ci_high = math.exp(coef + 1.96 * se)
    return {
        "coef": coef,
        "se": se,
        "p": p,
        "irr_full": irr_full,
        "irr_step": irr_step,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def group_rate_ratio(agg: pd.DataFrame, dark_thresh=0.75, light_thresh=0.25):
    dark = agg.loc[agg["skin_tone"] >= dark_thresh]
    light = agg.loc[agg["skin_tone"] <= light_thresh]

    dark_red = dark["total_red_cards"].sum()
    dark_games = dark["total_games"].sum()
    light_red = light["total_red_cards"].sum()
    light_games = light["total_games"].sum()

    dark_rate = dark_red / dark_games if dark_games > 0 else np.nan
    light_rate = light_red / light_games if light_games > 0 else np.nan

    # rate ratio and approximate CI using Poisson counts
    # handle zero counts with 0.5 continuity correction
    dark_red_cc = dark_red if dark_red > 0 else 0.5
    light_red_cc = light_red if light_red > 0 else 0.5
    rr = (dark_red_cc / dark_games) / (light_red_cc / light_games)
    se_log_rr = math.sqrt(1 / dark_red_cc + 1 / light_red_cc)
    ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
    ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)

    return {
        "n_dark": len(dark),
        "n_light": len(light),
        "dark_red": float(dark_red),
        "dark_games": float(dark_games),
        "light_red": float(light_red),
        "light_games": float(light_games),
        "dark_rate": float(dark_rate),
        "light_rate": float(light_rate),
        "rr": float(rr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def correlation_tests(agg: pd.DataFrame):
    # Spearman correlation between skin_tone and red rate
    from scipy.stats import spearmanr

    rho, p = spearmanr(agg["skin_tone"], agg["red_per_game"])
    return {"spearman_rho": float(rho), "p": float(p)}


def main():
    df = load_data()
    agg = aggregate_player_level(df)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_players_with_skin": int(agg.shape[0]),
        "skin_tone_mean": float(agg["skin_tone"].mean()),
        "skin_tone_std": float(agg["skin_tone"].std()),
        "overall_red_rate": float(agg["total_red_cards"].sum() / agg["total_games"].sum()),
    }

    pois = poisson_regression(agg)
    rr = group_rate_ratio(agg)
    corr = correlation_tests(agg)

    out = {
        "summary": summary,
        "poisson": pois,
        "group_rr": rr,
        "correlation": corr,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
