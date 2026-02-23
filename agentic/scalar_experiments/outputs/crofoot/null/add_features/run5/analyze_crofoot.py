from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "crofoot.csv"

    df = pd.read_csv(data_path)

    # Construct key predictors for the research question.
    # Relative group size: focal group size relative to other group.
    df["rel_group_size"] = df["n_focal"] / df["n_other"]

    # Contest location advantage: how much closer the focal group is to
    # the center of its home range than the other group (positive = focal closer).
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors for interpretability in the logistic model.
    for col in ["rel_group_size", "loc_advantage"]:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std

    y = df["win"]
    X = df[["rel_group_size_z", "loc_advantage_z"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()
    odds_ratios = np.exp(params)
    ci_or = np.exp(conf_int)

    print("n_obs", len(df))
    for predictor in ["rel_group_size_z", "loc_advantage_z"]:
        coef = params[predictor]
        pval = pvalues[predictor]
        or_val = odds_ratios[predictor]
        or_ci_low, or_ci_high = ci_or.loc[predictor]
        print(
            predictor,
            "coef",
            coef,
            "p",
            pval,
            "OR",
            or_val,
            "CI_low_OR",
            or_ci_low,
            "CI_high_OR",
            or_ci_high,
        )

    print("intercept", params["const"], "p", pvalues["const"])


if __name__ == "__main__":
    main()

