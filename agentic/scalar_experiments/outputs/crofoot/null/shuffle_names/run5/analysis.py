import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["win_focal"] = df["m_focal"]

    # Group sizes (total individuals)
    df["focal_size"] = df["f_other"]
    df["other_size"] = df["win"]

    # Relative group size: positive if focal group is larger
    df["rel_size_diff"] = df["focal_size"] - df["other_size"]
    df["rel_size_ratio"] = df["focal_size"] / df["other_size"]

    # Contest location: distance (m) from each group's home-range center
    # Metadata: m_other = distance of focal group from its home-range center
    #           n_focal = distance of other group from its home-range center
    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]

    # Home-advantage measure: positive when focal group is closer to its center
    df["home_advantage"] = df["dist_other_center"] - df["dist_focal_center"]

    # Standardize predictors for comparability
    for col in ["rel_size_diff", "home_advantage"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    print("Summary of key derived variables:\n")
    print(df[["win_focal", "rel_size_diff", "home_advantage"]].describe())
    print("\nCorrelation between predictors:\n")
    print(df[["rel_size_diff_z", "home_advantage_z"]].corr())

    # Logistic regression: probability focal group wins
    formula = "win_focal ~ rel_size_diff_z + home_advantage_z"
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    print("\nLogistic regression results:")
    print(result.summary())

    # Odds ratios and 95% CIs
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    print("\nOdds ratios with 95% confidence intervals:")
    for name in params.index:
        print(
            f"{name:20s} OR={odds_ratios[name]:6.3f} "
            f"95% CI=({conf_odds.loc[name, 0]:6.3f}, {conf_odds.loc[name, 1]:6.3f}) "
            f"p={result.pvalues[name]:.4f}"
        )


if __name__ == "__main__":
    main()

