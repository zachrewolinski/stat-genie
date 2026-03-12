import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_model(df, x_col, offset_col, use_nb_if_overdisp=True):
    exog = sm.add_constant(df[x_col])
    endog = df["redCards"]
    offset = df[offset_col]

    poisson = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset).fit()
    dispersion = poisson.deviance / poisson.df_resid if poisson.df_resid > 0 else np.nan

    if use_nb_if_overdisp and dispersion > 1.5:
        nb = sm.NegativeBinomial(endog, exog, loglike_method="nb2", offset=offset)
        nb_res = nb.fit(disp=False)
        return {
            "model": "NegativeBinomial",
            "res": nb_res,
            "dispersion": dispersion,
        }
    return {
        "model": "Poisson",
        "res": poisson,
        "dispersion": dispersion,
    }


def irr_and_ci(res, term):
    coef = res.params[term]
    conf = res.conf_int().loc[term]
    irr = float(np.exp(coef))
    ci_low = float(np.exp(conf[0]))
    ci_high = float(np.exp(conf[1]))
    pval = float(res.pvalues[term])
    return irr, ci_low, ci_high, pval


def main():
    df = pd.read_csv("soccer.csv")

    # Average skin tone rating across two raters
    df["skin"] = df[["rater1", "rater2"]].mean(axis=1)

    # Basic cleaning
    df = df[df["skin"].notna() & df["redCards"].notna() & df["games"].notna()]
    df = df[df["games"] > 0]

    df["log_games"] = np.log(df["games"])
    df["dark_binary"] = (df["skin"] > 0.5).astype(int)

    # Group rates (dark vs light based on >0.5 threshold)
    group = df.groupby("dark_binary")
    rates = (group["redCards"].sum() / group["games"].sum()).to_dict()
    counts = group["redCards"].sum().to_dict()
    exposure = group["games"].sum().to_dict()

    # Extreme groups for robustness (light <=0.25, dark >=0.75)
    extreme = df[(df["skin"] <= 0.25) | (df["skin"] >= 0.75)].copy()
    extreme["dark_binary"] = (extreme["skin"] >= 0.75).astype(int)
    extreme["log_games"] = np.log(extreme["games"])
    extreme_group = extreme.groupby("dark_binary")
    extreme_rates = (extreme_group["redCards"].sum() / extreme_group["games"].sum()).to_dict()
    extreme_counts = extreme_group["redCards"].sum().to_dict()
    extreme_exposure = extreme_group["games"].sum().to_dict()

    # Regression models
    cont_model = fit_model(df, "skin", "log_games")
    bin_model = fit_model(df, "dark_binary", "log_games")
    extreme_bin_model = fit_model(extreme, "dark_binary", "log_games")

    cont_term = "skin"
    bin_term = "dark_binary"

    cont_irr, cont_ci_low, cont_ci_high, cont_p = irr_and_ci(cont_model["res"], cont_term)
    bin_irr, bin_ci_low, bin_ci_high, bin_p = irr_and_ci(bin_model["res"], bin_term)
    extreme_bin_irr, extreme_bin_ci_low, extreme_bin_ci_high, extreme_bin_p = irr_and_ci(
        extreme_bin_model["res"], bin_term
    )

    results = {
        "n_rows": int(len(df)),
        "n_players_with_skin": int(df["playerShort"].nunique()),
        "overall_red_cards": float(df["redCards"].sum()),
        "overall_games": float(df["games"].sum()),
        "rate_per_game_overall": float(df["redCards"].sum() / df["games"].sum()),
        "group_rates_binary": {
            "light_rate": float(rates.get(0, np.nan)),
            "dark_rate": float(rates.get(1, np.nan)),
            "light_red_cards": float(counts.get(0, 0.0)),
            "dark_red_cards": float(counts.get(1, 0.0)),
            "light_games": float(exposure.get(0, 0.0)),
            "dark_games": float(exposure.get(1, 0.0)),
        },
        "extreme_group_rates": {
            "light_rate": float(extreme_rates.get(0, np.nan)),
            "dark_rate": float(extreme_rates.get(1, np.nan)),
            "light_red_cards": float(extreme_counts.get(0, 0.0)),
            "dark_red_cards": float(extreme_counts.get(1, 0.0)),
            "light_games": float(extreme_exposure.get(0, 0.0)),
            "dark_games": float(extreme_exposure.get(1, 0.0)),
            "n_rows_extreme": int(len(extreme)),
        },
        "continuous_skin_model": {
            "model": cont_model["model"],
            "dispersion": float(cont_model["dispersion"]),
            "irr_per_unit_skin": cont_irr,
            "ci_low": cont_ci_low,
            "ci_high": cont_ci_high,
            "p_value": cont_p,
        },
        "binary_skin_model": {
            "model": bin_model["model"],
            "dispersion": float(bin_model["dispersion"]),
            "irr_dark_vs_light": bin_irr,
            "ci_low": bin_ci_low,
            "ci_high": bin_ci_high,
            "p_value": bin_p,
        },
        "binary_skin_model_extreme": {
            "model": extreme_bin_model["model"],
            "dispersion": float(extreme_bin_model["dispersion"]),
            "irr_dark_vs_light": extreme_bin_irr,
            "ci_low": extreme_bin_ci_low,
            "ci_high": extreme_bin_ci_high,
            "p_value": extreme_bin_p,
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
