import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def poisson_irr(result, coef_name):
    coef = result.params[coef_name]
    se = result.bse[coef_name]
    irr = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    pval = result.pvalues[coef_name]
    return {
        "coef": float(coef),
        "irr": float(irr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pval": float(pval),
    }


def main():
    df = pd.read_csv("soccer.csv")
    df["skin"] = df[["rater1", "rater2"]].mean(axis=1)

    # Dyad-level analysis with exposure offset (games)
    dyad = df[(df["games"] > 0) & (~df["skin"].isna())].copy()
    dyad["dark"] = (dyad["skin"] > 0.5).astype(int)
    dyad["red_rate"] = dyad["redCards"] / dyad["games"]

    group_stats = (
        dyad.groupby("dark")
        .agg(
            dyads=("playerShort", "count"),
            total_red=("redCards", "sum"),
            total_games=("games", "sum"),
            mean_red_rate=("red_rate", "mean"),
            mean_skin=("skin", "mean"),
        )
        .reset_index()
    )

    # Poisson regression with offset for games; cluster-robust SE by player
    exog = sm.add_constant(dyad["dark"])
    model = sm.GLM(
        dyad["redCards"],
        exog,
        family=sm.families.Poisson(),
        offset=np.log(dyad["games"]),
    )
    res = model.fit(cov_type="cluster", cov_kwds={"groups": dyad["playerShort"]})

    irr_dark = poisson_irr(res, "dark")
    dispersion = float(res.deviance / res.df_resid)

    # Continuous skin tone model: effect per 0.25 step (one rating category)
    dyad["skin_step"] = dyad["skin"] / 0.25
    exog2 = sm.add_constant(dyad["skin_step"])
    model2 = sm.GLM(
        dyad["redCards"],
        exog2,
        family=sm.families.Poisson(),
        offset=np.log(dyad["games"]),
    )
    res2 = model2.fit(cov_type="cluster", cov_kwds={"groups": dyad["playerShort"]})
    irr_skin = poisson_irr(res2, "skin_step")

    output = {
        "dyad_level": {
            "n_dyads": int(dyad.shape[0]),
            "group_stats": group_stats.to_dict(orient="records"),
            "irr_dark": irr_dark,
            "dispersion": dispersion,
            "irr_skin_per_step": irr_skin,
        }
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
