import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Rename key columns to clearer semantic names based on info.json descriptions
    df = df.rename(
        columns={
            "m_focal": "win_focal",  # 1 if focal group won, 0 otherwise
            "m_other": "dist_focal_home",  # distance of focal group to its home-range center
            "n_focal": "dist_other_home",  # distance of other group to its home-range center
            "f_other": "size_focal",  # number of individuals in focal group
            "win": "size_other",  # number of individuals in other group
        }
    )

    # Drop any rows with missing values in the variables of interest (defensive, though none expected)
    cols_of_interest = [
        "win_focal",
        "dist_focal_home",
        "dist_other_home",
        "size_focal",
        "size_other",
    ]
    df = df.dropna(subset=cols_of_interest).copy()

    # Construct relative group-size and contest-location variables
    df["rel_group_size"] = df["size_focal"] / df["size_other"]
    df["log_rel_group_size"] = np.log(df["rel_group_size"])

    # Positive when contest is closer to focal group’s home range center
    df["dist_diff_other_minus_focal"] = (
        df["dist_other_home"] - df["dist_focal_home"]
    )
    df["focal_closer_to_home"] = (df["dist_focal_home"] < df["dist_other_home"]).astype(
        int
    )

    # Prepare design matrix for logistic regression:
    #   P(win_focal) = logit^{-1}(beta0 + beta1 * log_rel_group_size
    #                             + beta2 * focal_closer_to_home)
    y = df["win_focal"]
    X = df[["log_rel_group_size", "focal_closer_to_home"]]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X).fit(disp=False)

    print("Logistic regression results: win_focal ~ log_rel_group_size + focal_closer_to_home")
    print(model.summary())

    # Also print odds ratios and 95% confidence intervals
    params = model.params
    conf = model.conf_int()
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    print("\nOdds ratios (exp(coef)) with 95% CI:")
    for name in params.index:
        print(
            f"{name:24s} OR={odds_ratios[name]:6.3f}  "
            f"95% CI=({conf_or.loc[name, 0]:6.3f}, {conf_or.loc[name, 1]:6.3f})"
        )

    # Show simple descriptive stats for transparency
    print("\nDescriptive statistics:")
    print(
        df[["win_focal", "rel_group_size", "focal_closer_to_home"]].describe(
            include="all"
        )
    )

    # Simple group-wise summaries for additional intuition
    print("\nMean relative group size by outcome (win_focal):")
    print(df.groupby("win_focal")["rel_group_size"].mean())

    print("\nProportion of contests where focal is closer to home, by outcome:")
    print(df.groupby("win_focal")["focal_closer_to_home"].mean())


if __name__ == "__main__":
    main()
