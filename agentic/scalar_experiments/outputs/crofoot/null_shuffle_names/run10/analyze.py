import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group wins, 0 otherwise
    y = df["m_focal"]

    # According to metadata, these map to:
    # - f_other: number of individuals in focal group
    # - win:    number of individuals in other group
    # - m_other: distance of focal group from home-range center
    # - n_focal: distance of other group from home-range center
    focal_size = df["f_other"]
    other_size = df["win"]
    focal_dist = df["m_other"]
    other_dist = df["n_focal"]

    # Relative predictors
    df["size_diff"] = focal_size - other_size  # positive -> focal larger
    df["dist_diff"] = other_dist - focal_dist  # positive -> contest closer to focal

    X = df[["size_diff", "dist_diff"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    print("Logit coefficients:")
    print(model.params)
    print("\nStandard errors:")
    print(model.bse)
    print("\nP-values:")
    print(model.pvalues)
    print("\nPseudo R-squared (McFadden):", model.prsquared)


if __name__ == "__main__":
    main()

