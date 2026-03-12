import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Compute average skin rating across two raters
    df["skin_avg"] = df[["feature18", "feature19"]].mean(axis=1)

    # Keep rows with skin ratings
    df_skin = df[df["skin_avg"].notna()].copy()

    # Aggregate to player level
    player = (
        df_skin.groupby("feature1", as_index=False)
        .agg(
            skin_avg=("skin_avg", "mean"),
            games=("feature9", "sum"),
            red_cards=("feature16", "sum"),
        )
    )

    # Define light/dark groups using extreme categories
    light = player[player["skin_avg"] <= 0.25].copy()
    dark = player[player["skin_avg"] >= 0.75].copy()

    # Define broader groups around the midpoint to avoid sparse extremes
    light_broad = player[player["skin_avg"] < 0.5].copy()
    dark_broad = player[player["skin_avg"] > 0.5].copy()

    def rate_stats(sub):
        games = sub["games"].sum()
        reds = sub["red_cards"].sum()
        rate = reds / games if games > 0 else np.nan
        return games, reds, rate

    light_games, light_reds, light_rate = rate_stats(light)
    dark_games, dark_reds, dark_rate = rate_stats(dark)

    light_b_games, light_b_reds, light_b_rate = rate_stats(light_broad)
    dark_b_games, dark_b_reds, dark_b_rate = rate_stats(dark_broad)

    rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan

    # Poisson regression with offset (player-level) using continuous skin_avg
    player = player[player["games"] > 0].copy()
    X = sm.add_constant(player["skin_avg"])
    model_cont = sm.GLM(
        player["red_cards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(player["games"]),
    ).fit()

    # Poisson regression comparing dark vs light only (extremes)
    if not light.empty and not dark.empty:
        ld = pd.concat([light, dark], ignore_index=True)
        ld["dark"] = (ld["skin_avg"] >= 0.75).astype(int)
        X_ld = sm.add_constant(ld["dark"])
        model_ld = sm.GLM(
            ld["red_cards"],
            X_ld,
            family=sm.families.Poisson(),
            offset=np.log(ld["games"]),
        ).fit()
        coef_dark = model_ld.params["dark"]
        p_dark = model_ld.pvalues["dark"]
        rr_dark = float(np.exp(coef_dark))
    else:
        model_ld = None
        coef_dark = np.nan
        p_dark = np.nan
        rr_dark = np.nan

    # Poisson regression comparing broad dark vs broad light (excluding midpoint)
    if not light_broad.empty and not dark_broad.empty:
        ldb = pd.concat([light_broad, dark_broad], ignore_index=True)
        ldb["dark_broad"] = (ldb["skin_avg"] > 0.5).astype(int)
        X_ldb = sm.add_constant(ldb["dark_broad"])
        model_ldb = sm.GLM(
            ldb["red_cards"],
            X_ldb,
            family=sm.families.Poisson(),
            offset=np.log(ldb["games"]),
        ).fit()
        coef_dark_b = model_ldb.params["dark_broad"]
        p_dark_b = model_ldb.pvalues["dark_broad"]
        rr_dark_b = float(np.exp(coef_dark_b))
    else:
        model_ldb = None
        coef_dark_b = np.nan
        p_dark_b = np.nan
        rr_dark_b = np.nan

    summary = {
        "n_rows": int(len(df)),
        "n_rows_skin": int(len(df_skin)),
        "n_players": int(df["feature1"].nunique()),
        "n_players_skin": int(player["skin_avg"].notna().sum()),
        "skin_avg_counts": player["skin_avg"].round(2).value_counts().sort_index().to_dict(),
        "light_players": int(len(light)),
        "dark_players": int(len(dark)),
        "light_broad_players": int(len(light_broad)),
        "dark_broad_players": int(len(dark_broad)),
        "light_games": float(light_games),
        "light_reds": float(light_reds),
        "light_rate": float(light_rate),
        "dark_games": float(dark_games),
        "dark_reds": float(dark_reds),
        "dark_rate": float(dark_rate),
        "rate_ratio_dark_vs_light": float(rate_ratio),
        "light_broad_games": float(light_b_games),
        "light_broad_reds": float(light_b_reds),
        "light_broad_rate": float(light_b_rate),
        "dark_broad_games": float(dark_b_games),
        "dark_broad_reds": float(dark_b_reds),
        "dark_broad_rate": float(dark_b_rate),
        "rate_ratio_dark_vs_light_broad": float(dark_b_rate / light_b_rate) if light_b_rate > 0 else np.nan,
        "poisson_cont_coef_skin_avg": float(model_cont.params["skin_avg"]),
        "poisson_cont_p_skin_avg": float(model_cont.pvalues["skin_avg"]),
        "poisson_cont_rr_per_unit_skin": float(np.exp(model_cont.params["skin_avg"])),
        "poisson_dark_vs_light_rr": float(rr_dark),
        "poisson_dark_vs_light_p": float(p_dark),
        "poisson_dark_broad_rr": float(rr_dark_b),
        "poisson_dark_broad_p": float(p_dark_b),
    }

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
