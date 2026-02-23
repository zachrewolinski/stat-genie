import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")
    df["log_rel_size"] = np.log(df["n_focal"] / df["n_other"])
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    X = df[["log_rel_size", "focal_closer"]]
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    print(logit_model.summary())

    params = logit_model.params
    conf_int = logit_model.conf_int()
    odds_ratios = np.exp(params)
    or_ci = np.exp(conf_int)

    print("\nOdds ratios (exp(coef)) with 95% CI and p-values:")
    for name in ["log_rel_size", "focal_closer"]:
        print(
            f"{name}: OR={odds_ratios[name]:.3f}, "
            f"95% CI=({or_ci.loc[name, 0]:.3f}, {or_ci.loc[name, 1]:.3f}), "
            f"p={logit_model.pvalues[name]:.4f}"
        )


if __name__ == "__main__":
    main()

