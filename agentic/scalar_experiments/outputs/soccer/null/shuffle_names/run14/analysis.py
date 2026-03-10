import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Column semantics (from info.json):
    # rater1 and nExp are skin ratings (0-1). yellowCards is red card count.
    # redCards is number of games in the player-referee dyad (exposure).
    df["skin_mean"] = df[["rater1", "nExp"]].mean(axis=1)
    df["red_cards"] = df["yellowCards"].astype(float)
    df["games_exposure"] = df["redCards"].astype(float)

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["skin_mean", "red_cards", "games_exposure", "photoID"])
    df = df[df["games_exposure"] > 0]

    # Define skin groups: light (<0.5), dark (>0.5), drop neutral (==0.5)
    df["skin_group"] = np.where(
        df["skin_mean"] > 0.5,
        "dark",
        np.where(df["skin_mean"] < 0.5, "light", "medium"),
    )

    df_binary = df[df["skin_group"].isin(["dark", "light"])].copy()
    df_binary["dark"] = (df_binary["skin_group"] == "dark").astype(int)

    # Aggregate rate per game
    agg = (
        df_binary.groupby("skin_group")
        .agg(red_cards=("red_cards", "sum"), games=("games_exposure", "sum"), n=("red_cards", "size"))
        .reset_index()
    )
    agg["rate_per_game"] = agg["red_cards"] / agg["games"]

    # Poisson regression with offset (continuous skin)
    X_cont = sm.add_constant(df[["skin_mean"]])
    model_cont = sm.GLM(
        df["red_cards"],
        X_cont,
        family=sm.families.Poisson(),
        offset=np.log(df["games_exposure"]),
    )
    res_cont = model_cont.fit(cov_type="cluster", cov_kwds={"groups": df["photoID"]})

    # Poisson regression with offset (dark vs light)
    X_bin = sm.add_constant(df_binary[["dark"]])
    model_bin = sm.GLM(
        df_binary["red_cards"],
        X_bin,
        family=sm.families.Poisson(),
        offset=np.log(df_binary["games_exposure"]),
    )
    res_bin = model_bin.fit(cov_type="cluster", cov_kwds={"groups": df_binary["photoID"]})

    def irr_and_ci(res, key):
        coef = res.params[key]
        se = res.bse[key]
        irr = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
        pval = float(res.pvalues[key])
        return irr, ci_low, ci_high, pval

    irr_cont, ci_l_cont, ci_h_cont, p_cont = irr_and_ci(res_cont, "skin_mean")
    irr_bin, ci_l_bin, ci_h_bin, p_bin = irr_and_ci(res_bin, "dark")

    results = {
        "n_rows": int(df.shape[0]),
        "n_rows_binary": int(df_binary.shape[0]),
        "agg": agg.to_dict(orient="records"),
        "cont": {
            "irr": irr_cont,
            "ci_low": ci_l_cont,
            "ci_high": ci_h_cont,
            "p_value": p_cont,
        },
        "bin": {
            "irr": irr_bin,
            "ci_low": ci_l_bin,
            "ci_high": ci_h_bin,
            "p_value": p_bin,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
