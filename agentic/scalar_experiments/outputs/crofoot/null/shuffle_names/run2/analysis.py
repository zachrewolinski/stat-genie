import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Map columns to meaningful names based on info.json
    df["size_focal"] = df["f_other"]  # number of individuals in focal group
    df["size_other"] = df["win"]  # number of individuals in other group
    df["rel_size"] = df["size_focal"] - df["size_other"]

    df["dist_focal_center"] = df["m_other"]  # distance of focal group from its home-range center
    df["dist_other_center"] = df["n_focal"]  # distance of other group from its home-range center
    # Positive values => contest is relatively closer to focal group's home-range center
    df["loc_adv"] = df["dist_other_center"] - df["dist_focal_center"]

    df["focal_larger"] = (df["rel_size"] > 0).astype(int)
    df["focal_home_adv"] = (df["loc_adv"] > 0).astype(int)

    print("Dataset shape:", df.shape)
    print("\nProportion of focal wins by relative group size (focal larger vs not):")
    print(df.groupby("focal_larger")["m_focal"].mean())

    print("\nProportion of focal wins by location advantage (focal closer to its center vs not):")
    print(df.groupby("focal_home_adv")["m_focal"].mean())

    print("\nCross-tab of focal wins by size and location advantage:")
    print(
        df.groupby(["focal_larger", "focal_home_adv"])["m_focal"].agg(
            ["mean", "count"]
        )
    )

    # Logistic regression with clustered standard errors by dyad
    model = smf.glm(
        formula="m_focal ~ rel_size + loc_adv",
        data=df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    print("\nLogistic regression results (m_focal ~ rel_size + loc_adv):")
    print(model.summary())

    # Logistic regression using binary indicators for size and location advantages
    model_bin = smf.glm(
        formula="m_focal ~ focal_larger + focal_home_adv",
        data=df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    print(
        "\nLogistic regression results "
        "(m_focal ~ focal_larger + focal_home_adv):"
    )
    print(model_bin.summary())


if __name__ == "__main__":
    main()
