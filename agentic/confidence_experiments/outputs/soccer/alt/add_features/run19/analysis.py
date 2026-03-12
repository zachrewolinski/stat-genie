import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"


def fit_poisson(df, exposure_col, outcome_col, predictor_col, cluster_col):
    # Build design matrix with intercept and predictor
    X = sm.add_constant(df[predictor_col])
    y = df[outcome_col]
    offset = np.log(df[exposure_col])
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    # Cluster-robust SEs by player to account for repeated dyads
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return res


def summarize_group_rates(df, group_col, outcome_col, exposure_col):
    grouped = df.groupby(group_col, dropna=False).agg(
        red_cards=(outcome_col, "sum"),
        games=(exposure_col, "sum"),
        dyads=(outcome_col, "size"),
        players=("playerShort", "nunique"),
    )
    grouped["rate_per_game"] = grouped["red_cards"] / grouped["games"]
    return grouped.reset_index()


def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone average (0 to 1 scale). Require both raters to avoid missingness issues.
    df["skin_avg"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df[df["skin_avg"].notna()].copy()

    # Primary binary split: dark if strictly above neutral midpoint (0.5)
    df["dark"] = (df["skin_avg"] > 0.5).astype(int)

    # Summaries
    overall_rates = summarize_group_rates(df, "dark", "redCards", "games")

    # Poisson model with exposure
    res = fit_poisson(df, "games", "redCards", "dark", "playerShort")
    irr = float(np.exp(res.params["dark"]))
    pval = float(res.pvalues["dark"])
    coef = float(res.params["dark"])
    se = float(res.bse["dark"])

    # Sensitivity: extreme groups only
    extreme = df[(df["skin_avg"] <= 0.25) | (df["skin_avg"] >= 0.75)].copy()
    extreme["dark_extreme"] = (extreme["skin_avg"] >= 0.75).astype(int)
    extreme_rates = summarize_group_rates(extreme, "dark_extreme", "redCards", "games")
    if extreme["dark_extreme"].nunique() == 2:
        res_ext = fit_poisson(extreme, "games", "redCards", "dark_extreme", "playerShort")
        irr_ext = float(np.exp(res_ext.params["dark_extreme"]))
        pval_ext = float(res_ext.pvalues["dark_extreme"])
    else:
        irr_ext = np.nan
        pval_ext = np.nan

    # Output key stats for manual reasoning
    summary = {
        "n_rows": int(df.shape[0]),
        "n_players": int(df["playerShort"].nunique()),
        "overall_rates": overall_rates.to_dict(orient="records"),
        "poisson": {
            "coef": coef,
            "se": se,
            "irr": irr,
            "pval": pval,
        },
        "extreme_rates": extreme_rates.to_dict(orient="records"),
        "poisson_extreme": {
            "irr": irr_ext,
            "pval": pval_ext,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
