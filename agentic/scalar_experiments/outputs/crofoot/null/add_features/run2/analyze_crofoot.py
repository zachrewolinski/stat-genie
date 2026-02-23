import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    df = df.dropna(
        subset=["win", "n_focal", "n_other", "dist_focal", "dist_other"]
    ).copy()

    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["log_size_ratio"] = np.log(df["size_ratio"])
    df["loc_advantage"] = (df["dist_other"] - df["dist_focal"]) / (
        df["dist_other"] + df["dist_focal"]
    )

    df["rel_size_std"] = (df["rel_size"] - df["rel_size"].mean()) / df[
        "rel_size"
    ].std(ddof=0)
    df["rel_dist_std"] = (df["rel_dist"] - df["rel_dist"].mean()) / df[
        "rel_dist"
    ].std(ddof=0)

    model = smf.logit(
        "win ~ rel_size_std + rel_dist_std",
        data=df,
    ).fit(disp=False)

    print("Logistic regression of win on relative size and location")
    print(model.summary())

    params = model.params
    conf = model.conf_int()
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    print("\nOdds ratios (exp(beta)) with 95% CI and p-values:")
    for name in params.index:
        or_val = odds_ratios[name]
        ci_low, ci_high = conf_or.loc[name]
        p_val = model.pvalues[name]
        print(
            f"{name:15s} OR={or_val:6.3f} "
            f"95% CI=({ci_low:6.3f}, {ci_high:6.3f}) "
            f"p={p_val:7.4f}"
        )

    print("\nAlternative encoding: log(size ratio) and location advantage")
    model_alt = smf.logit(
        "win ~ log_size_ratio + loc_advantage",
        data=df,
    ).fit(disp=False)
    print(model_alt.summary())

    params_alt = model_alt.params
    conf_alt = model_alt.conf_int()
    odds_alt = np.exp(params_alt)
    conf_or_alt = np.exp(conf_alt)

    print("\nAlternative model odds ratios (exp(beta)) with 95% CI and p-values:")
    for name in params_alt.index:
        or_val = odds_alt[name]
        ci_low, ci_high = conf_or_alt.loc[name]
        p_val = model_alt.pvalues[name]
        print(
            f"{name:15s} OR={or_val:6.3f} "
            f"95% CI=({ci_low:6.3f}, {ci_high:6.3f}) "
            f"p={p_val:7.4f}"
        )


if __name__ == "__main__":
    main()
