import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Required columns
    red = df["feature16"].astype(float)
    games = df["feature9"].astype(float)
    skin1 = df["feature18"].astype(float)
    skin2 = df["feature19"].astype(float)

    skin_mean = pd.concat([skin1, skin2], axis=1).mean(axis=1, skipna=True)
    df = df.assign(
        red_cards=red,
        games=games,
        skin_mean=skin_mean,
    )

    # Drop rows without skin info or games <= 0
    df = df[(df["games"] > 0) & (df["skin_mean"].notna())]

    # Define skin groups: light (<0.5), mid (=0.5), dark (>0.5)
    df["skin_group"] = np.where(df["skin_mean"] > 0.5, "dark",
                         np.where(df["skin_mean"] < 0.5, "light", "mid"))

    # Main analysis: compare dark vs light, exclude mid
    df_main = df[df["skin_group"].isin(["dark", "light"])].copy()
    df_main["dark"] = (df_main["skin_group"] == "dark").astype(int)

    # Group rate summaries
    group_summary = df_main.groupby("skin_group").agg(
        total_red=("red_cards", "sum"),
        total_games=("games", "sum"),
        mean_red=("red_cards", "mean"),
        mean_games=("games", "mean"),
        n=("red_cards", "size"),
    )
    group_summary["rate_per_game"] = group_summary["total_red"] / group_summary["total_games"]

    # Poisson regression with offset log(games)
    y = df_main["red_cards"].values
    X = sm.add_constant(df_main["dark"].values)
    offset = np.log(df_main["games"].values)

    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit(cov_type="HC0")

    coef = poisson_res.params[1]
    se = poisson_res.bse[1]
    pval = poisson_res.pvalues[1]
    irr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Overdispersion check
    pearson_chi2 = poisson_res.pearson_chi2
    df_resid = poisson_res.df_resid
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    results = {
        "n_rows_total": int(len(df)),
        "n_rows_main": int(len(df_main)),
        "group_summary": group_summary.reset_index().to_dict(orient="records"),
        "poisson": {
            "coef_dark": float(coef),
            "se_dark": float(se),
            "pval_dark": float(pval),
            "irr_dark": irr,
            "irr_ci_low": ci_low,
            "irr_ci_high": ci_high,
            "dispersion": float(dispersion),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
