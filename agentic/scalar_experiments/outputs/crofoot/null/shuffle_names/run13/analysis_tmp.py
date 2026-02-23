import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata (not strictly needed for analysis, but useful for context/debugging)
    with open("info.json", "r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:", question)

    # Load dataset
    df = pd.read_csv("crofoot.csv")
    print("\nData shape:", df.shape)
    print("\nHead:\n", df.head())

    # According to metadata in info.json, columns map as follows:
    # m_focal: 1 if focal won contest (binary outcome)
    # f_other: number of individuals in focal group
    # win:    number of individuals in other group
    # m_other: distance (m) of focal group from center of its home range
    # n_focal: distance (m) of other group from center of its home range
    #
    # We construct predictors that capture relative group size and contest location advantage.

    df["group_size_focal"] = df["f_other"]
    df["group_size_other"] = df["win"]
    df["group_size_diff"] = df["group_size_focal"] - df["group_size_other"]

    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]
    # Positive value means focal group is closer to its own center than the other group is to its own.
    df["loc_advantage"] = df["dist_other_center"] - df["dist_focal_center"]

    print("\nSummary of constructed predictors:")
    print(df[["group_size_diff", "loc_advantage"]].describe())

    y = df["m_focal"]

    # Unstandardized predictors for interpretable odds ratios
    X = df[["group_size_diff", "loc_advantage"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("\nLogistic regression summary (unstandardized predictors):")
    print(result.summary())

    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    odds_ratios = np.exp(params)
    conf_int_or = np.exp(conf_int)

    print("\nOdds ratios with 95% CI:")
    for name in params.index:
        print(
            f"{name}: OR={odds_ratios[name]:.3f}, "
            f"95% CI=({conf_int_or.loc[name, 0]:.3f}, {conf_int_or.loc[name, 1]:.3f}), "
            f"p={pvalues[name]:.4f}"
        )

    print("\nPseudo R-squared (McFadden):", result.prsquared)


if __name__ == "__main__":
    main()

